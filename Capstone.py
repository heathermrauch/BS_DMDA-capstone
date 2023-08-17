from matplotlib import pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle
from scipy import stats as st
import math, numpy as np, os, pandas as pd, requests, seaborn as sns


### SETTINGS
URL = 'https://data.cms.gov/data-api/v1/'
UUID = '54426646-2108-48e7-a339-730dcfabbe9a'
ENDPOINT = f'dataset/{UUID}/data'
N = 13570
SIZE = 5000
COLUMNS = ['Brnd_Name', 'Gnrc_Name', 'Tot_Mftr', 'Mftr_Name', 'Tot_Spndng',
           'Tot_Dsg_Unts', 'Tot_Clms', 'Tot_Benes', 'Avg_Spnd_Per_Dsg_Unt_Wghtd', 
           'Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Bene', 'Outlier_Flag', 'Year']
YEARS = range(2016, 2021)
FLOATS = ['Tot_Spndng', 'Tot_Dsg_Unts', 'Tot_Clms', 'Tot_Benes', 'Avg_Spnd_Per_Dsg_Unt_Wghtd', 
          'Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Bene']
GROUPS = ['Brnd_Name', 'Gnrc_Name', 'Year']
PAIRS = [('Avg_Spnd_Per_Dsg_Unt_Wghtd', 'Avg_Spnd_Per_Dsg_Unt_Wghtd_min_unt'),
         ('Avg_Spnd_Per_Dsg_Unt_Wghtd', 'Avg_Spnd_Per_Dsg_Unt_Wghtd_min_clm'),
         ('Avg_Spnd_Per_Dsg_Unt_Wghtd', 'Avg_Spnd_Per_Dsg_Unt_Wghtd_min_ben'),
         ('Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Clm_min_unt'),
         ('Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Clm_min_clm'),
         ('Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Clm_min_ben'),
         ('Avg_Spnd_Per_Bene', 'Avg_Spnd_Per_Bene_min_unt'),
         ('Avg_Spnd_Per_Bene', 'Avg_Spnd_Per_Bene_min_clm'),
         ('Avg_Spnd_Per_Bene', 'Avg_Spnd_Per_Bene_min_ben')]


### FUNCTIONS
def get_data(url:str, endpoint:str, n:int, size:int):
    ''' Places a GET request to the specified endpoint. Uses n and size to determind the number
    of required batches to run. Returns a pandas dataframe of the normalized JSON response.
    Required Parameters:
    url (string): the url against which the get request should be placed
    endpoint (string): the specific API endpoint being queried
    n (integer): the number of total records to return
    size (integer): the number of records to return with each batch. This helps with API limits
    '''
    request_url = url + endpoint
    offset = 0
    loops = int(math.ceil(n / size))
    data = None
    for i in range(loops):
        params = {'offset': offset, 'size': size}
        response = requests.get(request_url, params)
        if data is None:
            data = response.json()
        else:
            data += response.json()
        print(f'Batch {i+1} of {loops} complete')
        if i+1 == loops:
            print('All data obtained')
        else:
            offset += size
    return pd.json_normalize(data)

def wide_to_long(df:pd.DataFrame, columns:list, years:list):
    ''' Transforms the data from one column per year per field to years in separate rows
    Required Parameters:
    df (pandas DataFrame): the data in wide format with years in separate columns
    columns (list): the final list of columns to return
    years (list): the list of expected year values
    '''
    print('Transforming data from wide to long format on year...')
    data = pd.DataFrame(columns=columns)
    for year in years:
        columns_year = [x.replace(str(min(years)), f'{year}') for x in df.columns
                            if x in ['Brnd_Name', 'Gnrc_Name', 'Tot_Mftr', 'Mftr_Name']
                                or x[-4:] == str(min(years))]
        data_year = df.loc[:, columns_year]
        data_year['Year'] = year
        data_year.columns = columns
        data = data.append(data_year)
    data = data.loc[data.Tot_Spndng.ne('') & data.Tot_Dsg_Unts.ne('') &
                    data.Tot_Clms.ne('') & data.Tot_Benes.ne('')]
    print('Transformation complete')
    return data

def float_columns(df:pd.DataFrame, columns:list):
    ''' Transforms a list of columns in a pandas DataFrame to floats
    Required Parameters:
    df (pandas DataFrame): DataFrame containing the columns to be transformed
    columns (list): the names of the columns to be transformed
    '''
    print('Converting numeric columns to floats...')
    for col in columns:
        df[col] = df[col].apply(float)
    print('Conversion complete')
    return df

def create_min_subset(df:pd.DataFrame, mins:pd.DataFrame, groups:list, field:str, suffix:str):
    ''' Creates a subset of the original data (df) using mins, groups, field, and suffix.
    Required Parameters:
    df (DataFrame): data to be subset
    mins (DataFrame): data containing minimum values
    groups (list): the groups used to aggregate mins. must also exist in df
    field (str): the field in mins used to determine the minimum
    suffix (str): the string to append to column titles in the data returned.
    '''
    print('Creating minimum subset...')
    right = mins[groups + [field]]
    data = df.merge(right, on=groups+[field]).drop('Tot_Mftr', axis=1)
    data.columns = ['Brnd_Name', 'Gnrc_Name', f'Mftr_Name_{suffix}', f'Tot_Spndg_{suffix}',
                    f'Tot_Dsg_Unts_{suffix}', f'Tot_Clms_{suffix}', f'Tot_Benes_{suffix}',
                    f'Avg_Spnd_Per_Dsg_Unt_Wghtd_{suffix}', f'Avg_Spnd_Per_Clm_{suffix}',
                    f'Avg_Spnd_Per_Bene_{suffix}', f'Outlier_Flag_{suffix}', 'Year']
    return data

def analysis(a:pd.Series, b:pd.Series, alpha:float, alternative:str):
    ''' Performs a paired t-test on a and b using the provided alpha and alternative hypothesis. 
    Both a and b must be the same length or the analysis will not be completed.
    Required Parameters:
    a (pd.Series): the first series to test
    b (pd.Series): the second series to test
    alpha (float): the value used to determine significance
    alternative (str): the alternative hypothesis to use. ('two-sided', 'less', 'greater')
    '''
    if len(a) == len(b):
        df = len(a) - 1
        critical_value = st.t.ppf(1-(alpha/2), df)
        tstat, pval = st.ttest_rel(a=a, b=b, alternative=alternative)
        diff = a - b
        mean_diff = np.mean(diff)
        std_err = st.sem(diff)
        confidence_interval = st.t.interval(alpha=1-alpha, loc=mean_diff, scale=std_err, df=df)
        print('t-critical:', critical_value,
              '\nt-statistic:', tstat, ', p-value:', pval,
              '\nmean difference:', mean_diff, ', standard error:', std_err,
              '\nconfidence interval:', confidence_interval)
        return (critical_value, tstat, pval, mean_diff, std_err, confidence_interval)
    else:
        print('a and b must be of equal length')


### FINAL APPLICATION
if __name__ == '__main__':
    
    # Get and pivot data
    df = get_data(URL, ENDPOINT, N, SIZE)
    data = wide_to_long(df, COLUMNS, YEARS)
    data = float_columns(data, FLOATS)
    
    # Split data into overall and per manufacturer spending
    overall = data[data.Mftr_Name.eq('Overall')]
    per_mftr = data[data.Mftr_Name.ne('Overall')]
   
    # Group data to find per manufacturer minimum values
    cols = GROUPS + ['Avg_Spnd_Per_Dsg_Unt_Wghtd', 'Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Bene']
    per_mftr_mins = per_mftr[cols].groupby(by=GROUPS, as_index=False).min()

    # Create the three subsets needed for testing
    per_mftr_min_unt = create_min_subset(per_mftr, per_mftr_mins, GROUPS,
                                         'Avg_Spnd_Per_Dsg_Unt_Wghtd', 'min_unt')
    per_mftr_min_clm = create_min_subset(per_mftr, per_mftr_mins, GROUPS,
                                         'Avg_Spnd_Per_Clm', 'min_clm')
    per_mftr_min_ben = create_min_subset(per_mftr, per_mftr_mins, GROUPS,
                                         'Avg_Spnd_Per_Bene', 'min_ben')

    ## Merge all three min datasets back to the 'Overall' dataset
    final = overall.merge(per_mftr_min_unt,
                          on=GROUPS).merge(per_mftr_min_clm,
                                           on=GROUPS).merge(per_mftr_min_ben, on=GROUPS)

    ## Save Final Data to .csv
    print('Saving final dataset...')
    final.to_csv('dataset.csv')
    print(f'Final dataset saved to {os.path.join(os.getcwd(), "dataset.csv")}')

    ## Boxplots for visual comparison
    print('Creating boxplots...')
    plt.rcParams['figure.figsize'] = 7.5, 2
    plt.rcParams['font.size'] = 8
    fig, ax = plt.subplots(1, 3, sharex=True, constrained_layout=True)
    for i, x in enumerate(['Avg_Spnd_Per_Dsg_Unt_Wghtd', 'Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Bene']):
        data = final[[x, f'{x}_min_unt', f'{x}_min_clm', f'{x}_min_ben']].melt()
        data.loc[data.variable.eq(x), 'label'] = 'overall'
        data.loc[data.variable.eq(f'{x}_min_unt'), 'label'] = 'min_unt'
        data.loc[data.variable.eq(f'{x}_min_clm'), 'label'] = 'min_clm'
        data.loc[data.variable.eq(f'{x}_min_ben'), 'label'] = 'min_ben'
        data.value = data.value.apply(float)
        sns.boxplot(x='label', y='value', data=data, showfliers=False, ax=ax[i])
        ax[i].set_title(x)
        if i == 1:
            ax[i].set_xlabel('method used to identify the manufacturer with the lowest average annual spending')
        else:
            ax[i].set_xlabel('')
        ax[i].set_ylabel('')
    plt.savefig('boxplots.png')
    print(f'Boxplots saved to {os.path.join(os.getcwd(), "boxplots.png")}')

    ## Zoomed boxplots
    print('Creating zoomed boxplots...')
    plt.rcParams['figure.figsize'] = 7.5, 6
    fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    for i, x in enumerate(['Avg_Spnd_Per_Dsg_Unt_Wghtd', 'Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Bene']):
        data = final[[x, f'{x}_min_unt', f'{x}_min_clm', f'{x}_min_ben']].melt()
        data.loc[data.variable.eq(x), 'label'] = 'overall'
        data.loc[data.variable.eq(f'{x}_min_unt'), 'label'] = 'min_unt'
        data.loc[data.variable.eq(f'{x}_min_clm'), 'label'] = 'min_clm'
        data.loc[data.variable.eq(f'{x}_min_ben'), 'label'] = 'min_ben'
        data.value = data.value.apply(float)
        sns.boxplot(x='label', y='value', data=data, showfliers=False, ax=ax[i])
        if i == 0:
            ax[i].set_ylim(0, 10)
            ax[i].set_xlabel('')
        elif i == 1:
            ax[i].set_ylim(0, 400)
            ax[i].set_xlabel('')
        elif i == 2:
            ax[i].set_ylim(0, 1000)
            ax[i].set_xlabel('method used to identify the manufacturer with the lowest average annual spending')
        ax[i].set_title(x)
        ax[i].set_ylabel('')
    plt.savefig('boxplots_zoom.png')
    print(f'Zoomed boxplots saved to {os.path.join(os.getcwd(), "boxplots_zoom.png")}')

    ## Perform analysis
    print('Performing analysis...')
    alpha = 0.05
    results = []
    for pair in PAIRS:
        results.append(analysis(final[pair[0]], final[pair[1]], alpha, 'greater'))
    print('Analysis complete')

    ## Plot t-distributions
    print('Creating t-distributions...')
    dof = len(final) - 1
    plt.rcParams['figure.figsize'] = 8, 6
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, constrained_layout=True)
    xs = np.linspace(-5, 13, 1000)
    i = 0
    for r, row in enumerate(axs):
        for c, col in enumerate(row):
            tcrit = np.round(results[i][0], 3)
            tstat = np.round(results[i][1], 3)
            axs[r,c].plot(xs, st.t.pdf(xs, dof), 'k')
            axs[r,c].vlines(tcrit, 0, st.t.pdf(tcrit, dof), 'r', label='t-critical')
            axs[r,c].text(tcrit, st.t.pdf(tcrit, dof), f'{tcrit}', color='r')
            axs[r,c].plot(tstat, st.t.pdf(tstat, dof), marker='o', label='t-score', color='b')
            axs[r,c].text(tstat-1, st.t.pdf(tstat, dof) + 0.02, f'{tstat}', color='b')
            axs[r,c].set_title(f'{PAIRS[i][0]} -\n{PAIRS[i][1]}', fontsize=8)
            if r == 2:
                axs[r,c].set_xlabel('t-score')
            if c == 0:
                axs[r,c].set_ylabel('Probability Density')
            axs[r,c].legend(loc='upper right')
            i += 1
    plt.savefig('t_dist.png')
    print(f't-distributions saved to {os.path.join(os.getcwd(), "t_dist.png")}')

    ## Produce PDF report
    print('Creating analysis report...')
    file_name = 'analysis_report.pdf'
    title_text = 'Reducing Prescription Drug Costs: An Analysis of Medicare Part D Spending by Drug \
    and Drug Manufacturer'
    introduction_text = 'The Medicare Part D Spending by Drug dataset is published on Data.CMS.gov \
    and contains "information on spending for drugs prescribed to Medicare beneficiaries enrolled in \
    Part D" including average spending per dosage unit, claim, and beneficiary by drug as well as by \
    drug and manufacturer. Below, the summary statistics are displayed for each variable.'
    summary1 = [['Variable', 'Mean', 'S.D.', 'Q1', 'Median', 'Q3']] + \
        [[x, np.round(np.mean(final.loc[:,x]), 3), np.round(np.std(final.loc[:,x], ddof=dof), 3),
          np.round(np.quantile(final.loc[:,x], .25), 3), np.round(np.quantile(final.loc[:,x], .5), 3),
          np.round(np.quantile(final.loc[:,x], .75), 3)] for x in ['Tot_Spndng', 'Tot_Dsg_Unts',
                                                                   'Tot_Clms', 'Tot_Benes',
                                                                   'Avg_Spnd_Per_Dsg_Unt_Wghtd',
                                                                   'Avg_Spnd_Per_Clm', 'Avg_Spnd_Per_Bene']]
    variables_text = 'Additionally, nine metrics were calculated to quantify the manufacturer of each \
    drug with the lowest average spending. The method used to identify the manufacturer with the lowest \
    average spending is appended to the end of each variable name in lower case letters. The options are \
    _min_unt, _min_clm, and _min_ben for Avg_Spnd_Per_Dsg_Unt_Wghtd, Avg_Spnd_Per_Clm, and Avg_Spnd_Per_Bene \
    respectively. The table below displays the summary statistics for the newly calcualted variables.'
    summary2 = [['Variable', 'Mean', 'S.D', 'Q1', 'Median', 'Q3']] + \
        [[x, np.round(np.mean(final.loc[:,x]), 3), np.round(np.std(final.loc[:,x], ddof=dof), 3),
          np.round(np.quantile(final.loc[:,x], .25), 3), np.round(np.quantile(final.loc[:,x], .5), 3),
          np.round(np.quantile(final.loc[:,x], .5), 3)] for x in ['Avg_Spnd_Per_Dsg_Unt_Wghtd_min_unt',
                                                                  'Avg_Spnd_Per_Dsg_Unt_Wghtd_min_clm',
                                                                  'Avg_Spnd_Per_Dsg_Unt_Wghtd_min_ben',
                                                                  'Avg_Spnd_Per_Clm_min_unt',
                                                                  'Avg_Spnd_Per_Clm_min_clm',
                                                                  'Avg_Spnd_Per_Clm_min_ben',
                                                                  'Avg_Spnd_Per_Bene_min_unt',
                                                                  'Avg_Spnd_Per_Bene_min_clm',
                                                                  'Avg_Spnd_Per_Bene_min_ben']]
    boxplots_text = 'The above figure displays each of the three original average spending metrics: \
    Avg_Spnd_Per_Dsg_Unt_Wghtd, Avg_Spnd_Per_Clm, and Avg_Spnd_Per_Bene; along with the same metric \
    calculated using the three minimum manufacturer calculations: min_unt, min_clm, and min_ben. Note \
    that statistical outliers, calcualted as values greater than or equal to quartile three plus 1.5 \
    times the Intra-Quartile Range, have been excluded to aid in visualization. Because of the wide \
    spread in each set, it is somewhat dificult to see that all three metrics are reduced when using \
    the manufacturer with the lowest annual average spending, regardless of the metric used to calculate \
    the lowest spending. The next set of figures displays the same three boxplots, zoomed in to to \
    better view the reduction in spending.'
    research_question = 'Is there a significant reduction in spending when prescriptions are only \
    sourced through drug manufacturers with the lowest average spending for each drug?'
    hypothesis = 'Prescription drug spending costs can be lowered by utilizing the drug manufacturer \
    associated with the lowest average annual prescription drug spending.'
    methodology = 'This analysis will use inferential statistics to test whether the differences \
    identified are significant. Specifically, a one-tailed (negative) paired t-test will be performed. \
    A paired t-test will be used to meet the assumption that the two groups are dependent on one \
    another. The test will be one-tailed in the negative direction because the minimum spending is, \
    by definition, lower than overall spending. To test the hypothesis, the following null-hypothesis \
    will be used: Mean drug spending for the manufacturer with the lowest associated annual spending will \
    be greater than or equal to overall mean spending. An alpha of 0.05 will be used to determine \
    significance. If the resulting p-value is below alpha, the null hypothesis will be rejected and the \
    alternate hypothesis, listed above, will be accepted.'
    result_text = 'The figure above displays the t-distribution resulting from this dataset along with the \
    critical t-score and actual t-scores for each test that was performed. It can be seen that using \
    any combination of the selected metrics renders a statistically significant difference in the mean \
    average annual spending by drug with t-scores ranging from 6.219 to 12.163. Full results are \
    displayed in the next table.'
    result_table = [['Variable', 'Method', 'Mean Difference/SE', 'Significance', '95% Confidence']] + \
        [[x[0], x[1][-7:], f'M={np.round(results[i][3], 3)}, SE={np.round(results[i][4], 3)}',
          f't({dof})={np.round(results[i][1], 3)}, p={results[i][2]}',
          f'[{np.round(results[i][5][0], 3)}, {np.round(results[i][5][1], 3)}]'] for i, x in enumerate(PAIRS)]
    conclusion = 'The t-test results indicate that, no matter which of the three methods are used to \
    identify the manufacturer with the lowest annual spending per drug, the difference is statistically \
    significant. The comparisons resulted in an average savings of $0.47 to $1.25 per dosage unit, $16.45 \
    to $32.68 per claim, and $133.39 to $274.27 per beneficiary for each drug annually. These results \
    suggest that there are missed opportunities for prescription drug savings within the Medicare Part D \
    program. If these savings were to be realized, this could result in a large reduction in government \
    spending on senior prescription drug coverage.'

    doc = SimpleDocTemplate(file_name, pagesize=letter)
    elements = list()

    # Title
    title_style = ParagraphStyle('normal', fontname='Helvetica-Bold', fontSize=16, leading=16, 
                                 alignment=1, spaceAfter=10)
    title = Paragraph(title_text, title_style)
    elements.append(title)

    # Introduction
    para_style = ParagraphStyle('Normal', spaceBefore=10, spaceAfter=10)
    para = Paragraph((introduction_text), para_style)
    elements.append(para)

    # Summary Table 1
    t_style = TableStyle([('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
                          ('BOX', (0,0), (-1, -1), 0.25, colors.black),
                          ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                          ('FONTSIZE', (0, 0), (-1, -1), 8)])
    t = Table(summary1, style=t_style)
    elements.append(t)

    # Variable Calculations
    para = Paragraph(variables_text, para_style)
    elements.append(para)

    # Summary Table 2
    t = Table(summary2, style=t_style)
    elements.append(t)

    # Boxplots
    elements.append(Paragraph('', ParagraphStyle('Normal', spaceBefore=10)))
    elements.append(Image('boxplots.png', width=450, height=120))

    # Boxplot Text
    para = Paragraph(boxplots_text, para_style)
    elements.append(para)

    # Boxplots Zoomed
    elements.append(Image('boxplots_zoom.png', width=450, height=360))

    # Question, Hypothesis, Methodology
    elements.append(Paragraph('Research Question:', 
                              ParagraphStyle('Normal', fontName='Helvetica-Bold', spaceBefore=10)))
    para = Paragraph(research_question, para_style)
    elements.append(para)
    elements.append(Paragraph('Hypothesis:', ParagraphStyle('Normal', fontName='Helvetica-Bold')))
    para = Paragraph(hypothesis, para_style)
    elements.append(para)
    elements.append(PageBreak())
    elements.append(Paragraph('Methodology:', ParagraphStyle('Normal', fontName='Helvetica-Bold')))
    para = Paragraph(methodology, para_style)
    elements.append(para)

    # Results
    elements.append(Paragraph('Results:', 
                              ParagraphStyle('Normal', fontName='Helvetica-Bold', spaceAfter=10)))
    elements.append(Image('t_dist.png', width=500, height=400))
    para = Paragraph(result_text, para_style)
    elements.append(para)
    elements.append(PageBreak())
    t = Table(result_table, style=t_style)
    elements.append(t)

    # Conclusion
    elements.append(Paragraph('Conclusion:',
                              ParagraphStyle('Normal', fontName='Helvetica-Bold', spaceBefore=10)))
    para = Paragraph(conclusion, para_style)
    elements.append(para)

    doc.build(elements)
    print(f'Analysis report saved to {os.path.join(os.getcwd(), "analysis_report.pdf")}')
    print('Program complete!')