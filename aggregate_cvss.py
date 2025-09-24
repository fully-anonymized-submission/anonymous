from pathlib import Path
import json
import math

def extract_cvss_scores(vulnerabilities, cwe_of_interest=['20', '22', '78', '79', '89', '502', '732', '798']):
    scores = {}
    for idx, vuln in enumerate(vulnerabilities):
        data_vuln = vuln['cve']
        metr = data_vuln['metrics']
        if 'cvssMetricV40' in metr:
            cvss = data_vuln['metrics']['cvssMetricV40']
            if len(cvss) > 1:
                continue
            cvss = data_vuln['metrics']['cvssMetricV40'][0]['cvssData']['baseScore']
        elif 'cvssMetricV31' in metr:
            cvss = data_vuln['metrics']['cvssMetricV31']
            if len(cvss) > 1:
                continue
            cvss = data_vuln['metrics']['cvssMetricV31'][0]['cvssData']['baseScore']

        if 'weaknesses' not in data_vuln:
            continue
        for cve in data_vuln['weaknesses']:
            for desc in cve['description']:
                cwe = desc['value']
            
                if cwe not in scores:
                    scores[cwe] = []
                    scores[cwe].append(cvss)
                else:
                    scores[cwe].append(cvss)
    cwe_of_interest = ['CWE-' + cwe for cwe in cwe_of_interest]
    scores = {k: v for k, v in scores.items() if k in cwe_of_interest}
    return scores
                
def aggregate_cvss(scores, base=2):
    """
    Aggregate CVSS scores per CWE using exponential-log averaging.

    Parameters
    ----------
    scores : dict
        Dictionary where keys are CWE identifiers (str)
        and values are lists of CVSS scores (float/int).
    base : int or float, optional
        Exponential base to use (default=2). 
        Use base=10 for steeper severity growth.

    Returns
    -------
    dict
        Dictionary with CWE identifiers as keys
        and aggregated CVSS scores (float) as values.
    """

    aggregated = {}

    for cwe, cvss_list in scores.items():
        if not cvss_list:
            aggregated[cwe] = 0.0
            continue

        total_weighted = sum(base ** cvss for cvss in cvss_list)
        avg_exp = total_weighted / len(cvss_list)
        aggregated[cwe] = math.log(avg_exp, base)

    return aggregated

def main():
    path_raw_scores = # INSERT PATH, AVAILABLE FROM https://nvd.nist.gov/vuln/data-feeds#divJson20Feeds
    path_to_save = # INSERT PATH TO SAVE AGGREGATED SCORES
    
    with open(path_raw_scores, 'r', encoding='utf-8') as file:
        data = json.load(file)

    vulnerabilities = data['vulnerabilities']
    scores = extract_cvss_scores(vulnerabilities)
    aggregated_scores = aggregate_cvss(scores, base=2)
    Path(path_to_save).parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_save, 'w', encoding='utf-8') as file:
        json.dump(aggregated_scores, file, indent=4)

if __name__ == "__main__":
    main()