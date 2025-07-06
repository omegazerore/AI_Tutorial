"""
Goal:
Help us extract the most relevant short- to mid-term trends (2025–2028) from a stack of PDF/Word reports for a strategy presentation. You will receive about 5 stacks of reports, each from a different source. Analyse each stack separately.

Your Task:
1.	Go through the reports and pull out only insights that are:
o	Relevant in the next 1–3 years
o	Actionable for cosnova (meaning from the color cosmetics & skincare segment)
2.	Group findings into 5 buckets:
o	Beauty Categories & Consumer Behavior
o	Sustainability
o	Societal Shifts
o	Market & Global Development
o	New Ways of Working
3.	Summarize each point in clear, concise bullet points. At the end, highlight the top 2–3 most important trends, why they matter and what action points cosnova should take.

Full Prompt (Summarized):
Go through the reports and extract only insights that are relevant within the next 1–3 years and actionable for cosnova, in the color cosmetics and skincare space. Organize your findings into five key buckets: Beauty Categories & Consumer Behavior, Sustainability, Societal Shifts, Market & Global Development, and New Ways of Working. At the end, highlight the top 2–3 most impactful trends, explain why they matter, and suggest concrete action points cosnova should consider.


Stack (71 files): https://cloud.cosnova.com/index.php/s/4R6f9sGjzS6hCd1

•	Beauty Categories & Consumer Trends: Growth or decline of specific categories, formats, or routines, Shifts in consumer behaviour or beauty expectations
•   Sustainability: True consumer demand for sustainability – by region or demographic, Relevance of standards and frameworks (e.g., SBTi)
•	Societal Shifts: Consumer and employee response to societal uncertainty or instability, Role and perception of DEI - including any backlash, Evolving expectations toward brands as trust in institutions erodes, The future of work and how we will work
•   Market & Global Development: Regional differences in opportunity (Europe, North America, Latin America, APAC, Middle East), Trends in distribution: e-commerce vs. offline, Where to invest or pull back
•   New Ways of Working: Strategic relevance of hybrid work, adaptability, continuous learning, Emerging leadership models
"""

import json
import logging
from collections import defaultdict

from tqdm import tqdm

from src.initialization import model_activation
from src.logic.trend_week_report.strategy_input import (
    megatrend_aggregation_pipeline,
    reference_matching_pipeline,
    summary_pipeline,
)
from src.logic.trend_week_report.strategy_input.report_writer import generate_trend_week_report

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrendWeekReport:
    """Handles the generation of a weekly trend report using LLM-based pipelines.

    This class loads notes from a JSON file, extracts and aggregates megatrends,
    summarizes insights and actions, matches references to source files,
    and generates a final formatted trend report.
    """
    def __init__(self, report_json_file: str, model: str):
        """Initializes the TrendWeekReport object.

        Args:
            report_json_file: Path to the JSON file containing trend notes.
            model: Name of the model to activate for pipeline processing.
        """
        self.report_json_file = report_json_file

        self.model = model_activation(model_name=model)

        self._load_notes()
        self._megatrend_aggregation_pipeline = self._build_pipeline(megatrend_aggregation_pipeline)
        self._reference_matching_pipeline = self._build_pipeline(reference_matching_pipeline)
        self._summary_pipeline = self._build_pipeline(summary_pipeline)

    def _load_notes(self):
        """Loads note data from the provided JSON file."""
        try:
            with open(self.report_json_file, 'r', encoding='utf-8') as f:
                self.notes_data = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"JSON file not found: {self.report_json_file}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in {self.report_json_file}")
            raise e
        except Exception as e:
            logger.exception(f"Unexpected error loading notes from {self.report_json_file}: {e}")
            raise

    def _build_pipeline(self, pipeline_module):
        """Builds a LangChain pipeline using the provided module.

        Args:
            pipeline_module: A module that defines a build_prompt_template and output_parser.

        Returns:
            A pipeline chain of prompt -> model -> parser.
        """

        prompt_template = pipeline_module.build_prompt_template()
        output_parser = pipeline_module.output_parser

        return prompt_template | self.model | output_parser

    def _megatrend_aggregation(self):
        """Aggregates insights and actions under megatrends from the raw note data."""

        logger.info("Start megatrend aggregation")

        self._megatrend_dict = {}

        for filename, note_data in tqdm(self.notes_data.items()):
            logger.info(f"{filename}")
            trends = note_data['trends']
            for trend in trends:
                # split for the time being
                megatrend = trend['name'].split(":")[0]
                actions = [a['name'] for a in trend['actions']]
                minireport = f"**Insight**:\n  {trend['insight']}\n**Actions**:\n"
                for action in actions:
                    minireport += f"- {action}\n"
                minireport += "\n"
                if megatrend not in self._megatrend_dict:
                    self._megatrend_dict[megatrend] = [minireport]
                else:
                    self._megatrend_dict[megatrend].append(minireport)

        logger.info(f"All the megatrends: {self._megatrend_dict.keys()}")

    def _prepare_megatrend_context(self) -> dict:
        """Concatenates all minireports per megatrend into one context string.

        Returns:
            Dictionary mapping megatrends to their combined context text.
        """
        megatrend_context = {megatrend: "".join(minireports) for megatrend, minireports in self._megatrend_dict.items()}

        return megatrend_context

    def _summarize_megatrends(self, megatrend_context: dict) -> dict:
        """Summarizes each megatrend using the aggregation pipeline.

        Args:
            megatrend_context: Dictionary of formatted megatrend content.

        Returns:
            A dictionary mapping megatrends to summarized output.
        """
        summaries = {}
        for megatrend, text in tqdm(megatrend_context.items()):
            logger.info(f"Megatrend: {megatrend} fusion")
            result = self._megatrend_aggregation_pipeline.invoke({"text": text})
            summaries[megatrend] = result

        return summaries

    def _match_references(self, summaries: dict):
        """Matches subtrends to source documents using reference matching.

        Args:
            summaries: Dictionary of summarized megatrend outputs.

        Returns:
            subtrend_2_reference: Mapping of subtrend names to reference filenames.
            doc_index_map: Mapping of filenames to bibliography indices.
            summary_batches: Raw batch data used during matching.
        """

        subtrend_2_reference = defaultdict(list)

        # document_index_2_appearance_order_hash_map: document index -> appearance order
        doc_index_map = {}
        reference_index = 0
        summary_batches = []

        for megatrend, summary in summaries.items():
            for note_idx, (filename, note_data) in enumerate(self.notes_data.items()):
                for trend_entry in note_data['trends']:
                    if trend_entry['name'] == megatrend:
                        break
                else:
                    continue

                note = trend_entry['insight']

                for key_finding in summary.key_findings:
                    summary_batches.append({"note": note, "subtrend": key_finding.name, "note_idx": note_idx,
                                            "filename": filename})

        outputs = self._reference_matching_pipeline.batch(summary_batches)

        for output, row in zip(outputs, summary_batches):
            name = row["filename"]
            if output['result'] == 'YES':
                subtrend_2_reference[row['subtrend']].append(name)
                if name not in doc_index_map:
                    doc_index_map[name] = reference_index
                    reference_index += 1

        return subtrend_2_reference, doc_index_map, summary_batches

    def _generate_summary_input_text(self, summaries: dict) -> str:
        """Formats the summaries into a final structured text string.

        Args:
            summaries: Dictionary of summarized megatrend outputs.

        Returns:
            A single formatted string to be passed into the summary pipeline.
        """
        lines = []

        for megatrend, summary in summaries.items():
            lines.append(f"*{megatrend}*:\n\n**Insight**:\n  {summary.insight}\n")
            lines.append("**Key Findings**:")
            lines.extend(f"- {kf.name}" for kf in summary.key_findings)
            lines.append("\n**Recommended Actions**:")
            lines.extend(f"- {action.name}" for action in summary.result)
            lines.append("")

        return "\n".join(lines)

    def run(self):
        """Main method to run the full report generation pipeline.

        Returns:
            A string representing the final rendered trend week report.
        """
        self._megatrend_aggregation() # Aggregate the megatrend content from various sourc

        context = self._prepare_megatrend_context()
        summaries = self._summarize_megatrends(context)
        subtrend_refs, doc_idx_map, summary_batches = self._match_references(summaries)
        summary_input_text = self._generate_summary_input_text(summaries)

        summary = self._summary_pipeline.invoke({"text": summary_input_text})

        final_doc = generate_trend_week_report(context, summaries, subtrend_refs, doc_idx_map, summary)

        return final_doc


if __name__ == "__main__":

    import os
    from datetime import datetime

    from src.io.path_definition import get_datafetch


    time_start = datetime.now()

    report_json_file = os.path.join(get_datafetch(), "Stylus", "report_raw.json")

    report_generator = TrendWeekReport(report_json_file=report_json_file,
                                       model="gpt-4.1-2025-04-14")

    final_doc = report_generator.run()

    final_doc.save(os.path.join(get_datafetch(), "Stylus", "report_raw.docx"))

    time_end = datetime.now()

    print(f"Time eclapse: {time_end - time_start}")