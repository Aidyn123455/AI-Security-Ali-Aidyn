import io
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "-q",
        "pandas>=2.1", "numpy>=1.26", "scikit-learn>=1.4",
        "matplotlib>=3.8", "seaborn>=0.13", "streamlit>=1.40",
        "joblib>=1.3", "pytest>=8.0",
    ],
    check=False,
)

WORK_ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
PROJECT_ROOT = WORK_ROOT / "ai_security_monitor"
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

EMBEDDED_PROJECT_FILES = {
  "scripts/__init__.py": "",
  "config.py": "from pathlib import Path\n\nBASE_DIR = Path(__file__).resolve().parent\nDATA_DIR = BASE_DIR / \"data\"\nMODELS_DIR = BASE_DIR / \"models\"\nOUTPUTS_DIR = BASE_DIR / \"outputs\"\nREPORTS_DIR = OUTPUTS_DIR / \"reports\"\n\nNETWORK_DATA_PATH = DATA_DIR / \"network_traffic.csv\"\nTEXT_DATA_PATH = DATA_DIR / \"http_requests.csv\"\n\nISOFOREST_MODEL_PATH = MODELS_DIR / \"network_isoforest.joblib\"\nRF_MODEL_PATH = MODELS_DIR / \"network_random_forest.joblib\"\nNLP_MODEL_PATH = MODELS_DIR / \"nlp_logreg.joblib\"\nNLP_NB_MODEL_PATH = MODELS_DIR / \"nlp_naive_bayes.joblib\"\n\nNETWORK_EVENTS_PATH = OUTPUTS_DIR / \"network_events.csv\"\nHTTP_EVENTS_PATH = OUTPUTS_DIR / \"http_events.csv\"\nMETRICS_PATH = REPORTS_DIR / \"metrics.json\"\n\nRANDOM_STATE = 42\nTEST_SIZE = 0.25\n\nNETWORK_LABEL_COL = \"Label\"\nNETWORK_TIME_COL = \"Timestamp\"\nNETWORK_BENIGN_LABEL = \"BENIGN\"\n\nNETWORK_FEATURES = [\n    \"Destination Port\",\n    \"Flow Duration\",\n    \"Total Fwd Packets\",\n    \"Total Backward Packets\",\n    \"Total Length of Fwd Packets\",\n    \"Total Length of Bwd Packets\",\n    \"Fwd Packet Length Mean\",\n    \"Bwd Packet Length Mean\",\n    \"Flow Bytes/s\",\n    \"Flow Packets/s\",\n    \"Flow IAT Mean\",\n    \"SYN Flag Count\",\n    \"ACK Flag Count\",\n    \"Average Packet Size\",\n]\n\nISOFOREST_CONTAMINATION = 0.05\n\nTEXT_COL = \"request\"\nTEXT_LABEL_COL = \"label\"\nTEXT_TIME_COL = \"timestamp\"\nTEXT_CLASSES = [\"Normal\", \"SQL Injection\", \"XSS\"]\nMAX_REQUEST_LENGTH = 4096\n\nNLP_REVIEW_THRESHOLD = 0.60\n\nEVENT_COLUMNS = [\n    \"event_id\",\n    \"timestamp\",\n    \"source_module\",\n    \"model_name\",\n    \"prediction\",\n    \"score\",\n    \"is_alert\",\n    \"needs_review\",\n    \"ground_truth\",\n    \"details\",\n    \"status\",\n]\nPREDICTION_STATUS = \"Прогноз модели (не подтверждено)\"\n\nMAX_UPLOAD_MB = 20\nMAX_UPLOAD_ROWS = 200_000\nALLOWED_EXTENSIONS = {\".csv\"}\n",
  "requirements.txt": "pandas>=2.1\nnumpy>=1.26\nscikit-learn>=1.4\nmatplotlib>=3.8\nseaborn>=0.13\nstreamlit>=1.40\njoblib>=1.3\npytest>=8.0\njupyter>=1.0\n",
  "pytest.ini": "[pytest]\ntestpaths = tests\npythonpath = .\n",
  "train_all.py": "from __future__ import annotations\n\nimport argparse\nimport json\n\nimport joblib\n\nimport config\nfrom src.common.io_utils import save_events\nfrom src.common.metrics import plot_confusion_matrix\n\ndef run_network() -> dict:\n    from src.network.models import analyze_network, network_events, train_network_models\n    from src.network.preprocess import clean_network_data, load_network_csv\n\n    print(\"== Модуль 1: сетевой трафик ==\")\n    df = load_network_csv(config.NETWORK_DATA_PATH, require_label=True)\n    df, quality = clean_network_data(df)\n    print(f\"Качество данных: {quality['duplicates']} дубликатов, \"\n          f\"{quality['missing_total']} пропусков, {quality['infinite_values']} inf\")\n\n    models, metrics = train_network_models(df)\n    joblib.dump(models[\"isoforest\"], config.ISOFOREST_MODEL_PATH)\n    joblib.dump(models[\"random_forest\"], config.RF_MODEL_PATH)\n\n    for key, title in [(\"isolation_forest\", \"Isolation Forest\"), (\"random_forest\", \"Random Forest\")]:\n        m = metrics[key]\n        plot_confusion_matrix(m[\"confusion_matrix\"], m[\"labels\"], title,\n                              config.REPORTS_DIR / f\"cm_network_{key}.png\")\n        print(f\"{title}: accuracy={m['accuracy']} f1_macro={m['f1_macro']} FPR={m['false_positive_rate']}\")\n\n    analyzed = analyze_network(df, models[\"isoforest\"], models[\"random_forest\"])\n    save_events(network_events(analyzed), config.NETWORK_EVENTS_PATH)\n    metrics[\"data_quality\"] = quality\n    return metrics\n\ndef run_nlp() -> dict:\n    from src.nlp.models import classify_requests, http_events, train_nlp_models\n    from src.nlp.preprocess import clean_text_data, load_text_csv\n\n    print(\"== Модуль 2: HTTP-запросы (NLP) ==\")\n    df = load_text_csv(config.TEXT_DATA_PATH, require_label=True)\n    df, quality = clean_text_data(df)\n    print(f\"После очистки: {quality['rows_after_cleaning']} строк, \"\n          f\"удалено дубликатов: {quality['duplicates_removed']}\")\n\n    models, metrics = train_nlp_models(df)\n    joblib.dump(models[\"logistic_regression\"], config.NLP_MODEL_PATH)\n    joblib.dump(models[\"naive_bayes\"], config.NLP_NB_MODEL_PATH)\n\n    for key, title in [(\"logistic_regression\", \"Logistic Regression\"), (\"naive_bayes\", \"Naive Bayes\")]:\n        m = metrics[key]\n        plot_confusion_matrix(m[\"confusion_matrix\"], m[\"labels\"], title,\n                              config.REPORTS_DIR / f\"cm_nlp_{key}.png\")\n        print(f\"{title}: accuracy={m['accuracy']} f1_macro={m['f1_macro']} FPR={m['false_positive_rate']}\")\n\n    classified = classify_requests(models[\"logistic_regression\"], df[config.TEXT_COL])\n    save_events(http_events(df, classified), config.HTTP_EVENTS_PATH)\n    metrics[\"data_quality\"] = quality\n    return metrics\n\ndef main() -> None:\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\"--only\", choices=[\"network\", \"nlp\"])\n    args = parser.parse_args()\n\n    for d in (config.MODELS_DIR, config.OUTPUTS_DIR, config.REPORTS_DIR):\n        d.mkdir(parents=True, exist_ok=True)\n\n    all_metrics = {}\n    if config.METRICS_PATH.exists():\n        all_metrics = json.loads(config.METRICS_PATH.read_text(encoding=\"utf-8\"))\n\n    if args.only in (None, \"network\"):\n        if config.NETWORK_DATA_PATH.exists():\n            all_metrics[\"network\"] = run_network()\n        else:\n            print(f\"Нет {config.NETWORK_DATA_PATH} — запустите scripts/generate_network_data.py\")\n    if args.only in (None, \"nlp\"):\n        if config.TEXT_DATA_PATH.exists():\n            all_metrics[\"nlp\"] = run_nlp()\n        else:\n            print(f\"Нет {config.TEXT_DATA_PATH} — подготовьте датасет: scripts/prepare_text_dataset.py\")\n\n    config.METRICS_PATH.write_text(\n        json.dumps(all_metrics, ensure_ascii=False, indent=2, default=str), encoding=\"utf-8\")\n    print(\"Метрики сохранены в\", config.METRICS_PATH)\n\nif __name__ == \"__main__\":\n    main()\n",
  "app.py": "from __future__ import annotations\n\nimport json\n\nimport joblib\nimport matplotlib\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\nimport streamlit as st\n\nimport config\nfrom src.common.io_utils import DataValidationError, load_events, to_csv_bytes\n\nst.set_page_config(page_title=\"AI Security Monitor\", page_icon=\"🛡️\", layout=\"wide\")\nsns.set_theme(style=\"whitegrid\")\n\n@st.cache_resource\ndef load_model(path_str: str):\n    from pathlib import Path\n    path = Path(path_str)\n\n    if path.parent.resolve() != config.MODELS_DIR.resolve() or not path.exists():\n        return None\n    return joblib.load(path)\n\ndef get_events(module: str) -> pd.DataFrame:\n    key = f\"{module}_events\"\n    if key not in st.session_state:\n        path = config.NETWORK_EVENTS_PATH if module == \"network\" else config.HTTP_EVENTS_PATH\n        st.session_state[key] = load_events(path)\n    return st.session_state[key]\n\ndef set_events(module: str, df: pd.DataFrame) -> None:\n    st.session_state[f\"{module}_events\"] = df\n\niso_model = load_model(str(config.ISOFOREST_MODEL_PATH))\nrf_model = load_model(str(config.RF_MODEL_PATH))\nnlp_model = load_model(str(config.NLP_MODEL_PATH))\n\nst.title(\"🛡️ AI Security Monitor\")\nst.caption(\"Учебный прототип: мониторинг сетевой активности и NLP-классификация HTTP-запросов. \"\n           \"Все результаты — прогнозы ML-моделей и требуют проверки аналитиком.\")\n\ntab_overview, tab_net, tab_http, tab_eval, tab_about = st.tabs(\n    [\"📊 Обзор\", \"🌐 Сетевой трафик\", \"🔤 HTTP-запросы (NLP)\", \"📈 Качество моделей\", \"ℹ️ О системе\"])\n\nwith tab_overview:\n    net = get_events(\"network\")\n    http = get_events(\"http\")\n    events = pd.concat([net, http], ignore_index=True)\n\n    c1, c2, c3, c4, c5 = st.columns(5)\n    c1.metric(\"Сетевых записей\", f\"{len(net):,}\".replace(\",\", \" \"))\n    c2.metric(\"Аномалий (IF)\", int(net[\"is_alert\"].sum()) if len(net) else 0)\n    c3.metric(\"Запросов Normal\", int((http[\"prediction\"] == \"Normal\").sum()) if len(http) else 0)\n    c4.metric(\"SQL Injection\", int((http[\"prediction\"] == \"SQL Injection\").sum()) if len(http) else 0)\n    c5.metric(\"XSS\", int((http[\"prediction\"] == \"XSS\").sum()) if len(http) else 0)\n\n    if events.empty:\n        st.info(\"Событий пока нет. Обучите модели (`python train_all.py`) или загрузите CSV \"\n                \"на вкладках «Сетевой трафик» и «HTTP-запросы».\")\n    else:\n        st.markdown(\"##### Распределение событий по категориям (прогноз моделей)\")\n        g1, g2 = st.columns(2)\n        for col, module, title in [(g1, \"network\", \"Сетевой модуль\"), (g2, \"http\", \"HTTP-модуль\")]:\n            sub = events[events[\"source_module\"] == module]\n            if sub.empty:\n                col.caption(f\"{title}: нет данных\")\n                continue\n            fig, ax = plt.subplots(figsize=(5, 2.8))\n            counts = sub[\"prediction\"].value_counts()\n            sns.barplot(x=counts.values, y=counts.index, ax=ax, color=\"#4C72B0\")\n            ax.set_title(title); ax.set_xlabel(\"Количество\"); ax.set_ylabel(\"\")\n            fig.tight_layout(); col.pyplot(fig); plt.close(fig)\n\n        st.markdown(\"##### Журнал событий\")\n        f1, f2, f3, f4 = st.columns([1, 1.4, 1, 1.6])\n        modules = f1.multiselect(\"Модуль\", [\"network\", \"http\"], default=[\"network\", \"http\"])\n        cats = f2.multiselect(\"Категория\", sorted(events[\"prediction\"].unique()))\n        result = f3.selectbox(\"Результат\", [\"Все\", \"Только предупреждения\", \"Требуют проверки\"])\n        ts = pd.to_datetime(events[\"timestamp\"], errors=\"coerce\")\n        time_range = None\n        if ts.notna().any():\n            lo, hi = ts.min().to_pydatetime(), ts.max().to_pydatetime()\n            if lo < hi:\n                time_range = f4.slider(\"Время\", min_value=lo, max_value=hi, value=(lo, hi),\n                                       format=\"DD.MM HH:mm\")\n\n        view = events[events[\"source_module\"].isin(modules)]\n        if cats:\n            view = view[view[\"prediction\"].isin(cats)]\n        if result == \"Только предупреждения\":\n            view = view[view[\"is_alert\"]]\n        elif result == \"Требуют проверки\":\n            view = view[view[\"needs_review\"]]\n        if time_range:\n            vts = pd.to_datetime(view[\"timestamp\"], errors=\"coerce\")\n            view = view[vts.between(*time_range) | vts.isna()]\n\n        view = view.sort_values(\"timestamp\", ascending=False)\n        st.caption(f\"Показано {min(len(view), 500)} из {len(view)} событий (сначала последние).\")\n        st.dataframe(view.head(500), width=\"stretch\", hide_index=True)\n        st.download_button(\"⬇️ Выгрузить отфильтрованные события (CSV)\", to_csv_bytes(view),\n                           \"events_filtered.csv\", \"text/csv\")\n\nwith tab_net:\n    from src.network.models import analyze_network, network_events\n    from src.network.preprocess import clean_network_data, load_network_csv\n\n    st.subheader(\"Мониторинг сетевой активности\")\n    if iso_model is None:\n        st.warning(\"Модель Isolation Forest не найдена. Выполните `python train_all.py --only network`.\")\n    else:\n        up = st.file_uploader(\"Загрузите CSV с сетевыми потоками (столбцы CIC-IDS2017)\",\n                              type=[\"csv\"], key=\"net_upload\",\n                              help=f\"До {config.MAX_UPLOAD_MB} МБ. Обязательные столбцы: \"\n                                   + \", \".join(config.NETWORK_FEATURES))\n        if up is not None and st.button(\"Проанализировать сетевой файл\"):\n            try:\n                raw = load_network_csv(up, filename=up.name)\n                clean, quality = clean_network_data(raw)\n                analyzed = analyze_network(clean, iso_model, rf_model)\n                st.session_state[\"net_analyzed\"] = analyzed\n                st.session_state[\"net_quality\"] = quality\n                set_events(\"network\", network_events(analyzed))\n                st.success(f\"Проанализировано {len(analyzed)} записей.\")\n            except DataValidationError as exc:\n                st.error(f\"Файл отклонён: {exc}\")\n\n        analyzed = st.session_state.get(\"net_analyzed\")\n        if analyzed is None and config.NETWORK_DATA_PATH.exists():\n            if st.button(\"Показать анализ обучающего набора\"):\n                clean, quality = clean_network_data(load_network_csv(config.NETWORK_DATA_PATH))\n                st.session_state[\"net_analyzed\"] = analyze_network(clean, iso_model, rf_model)\n                st.session_state[\"net_quality\"] = quality\n                st.rerun()\n\n        if analyzed is not None:\n            q = st.session_state.get(\"net_quality\", {})\n            a, b, c, d = st.columns(4)\n            a.metric(\"Строк до очистки\", q.get(\"rows\", \"—\"))\n            b.metric(\"Удалено дубликатов\", q.get(\"duplicates_removed\", \"—\"))\n            c.metric(\"Пропусков\", q.get(\"missing_total\", \"—\"))\n            d.metric(\"Значений inf\", q.get(\"infinite_values\", \"—\"))\n\n            g1, g2 = st.columns(2)\n            if config.NETWORK_LABEL_COL in analyzed.columns:\n                fig, ax = plt.subplots(figsize=(5, 3))\n                vc = analyzed[config.NETWORK_LABEL_COL].value_counts()\n                sns.barplot(x=vc.values, y=vc.index, ax=ax, color=\"#55A868\")\n                ax.set_title(\"Метки в данных (фактические)\"); ax.set_xlabel(\"Записей\")\n                fig.tight_layout(); g1.pyplot(fig); plt.close(fig)\n            fig, ax = plt.subplots(figsize=(5, 3))\n            sns.histplot(data=analyzed, x=\"if_score\", hue=\"if_anomaly\", bins=40, ax=ax,\n                         palette={False: \"#4C72B0\", True: \"#C44E52\"})\n            ax.set_title(\"Оценка аномальности (Isolation Forest)\"); ax.set_xlabel(\"if_score\")\n            fig.tight_layout(); g2.pyplot(fig); plt.close(fig)\n\n            fig, ax = plt.subplots(figsize=(10, 3.2))\n            sns.scatterplot(data=analyzed.sample(min(3000, len(analyzed)), random_state=1),\n                            x=\"Flow Duration\", y=\"Flow Packets/s\", hue=\"if_anomaly\", s=12, ax=ax,\n                            palette={False: \"#4C72B0\", True: \"#C44E52\"})\n            ax.set_xscale(\"symlog\"); ax.set_yscale(\"symlog\")\n            ax.set_title(\"Потоки: длительность vs пакетов/с (красные — аномалии по прогнозу)\")\n            fig.tight_layout(); st.pyplot(fig); plt.close(fig)\n\n            st.markdown(\"##### Записи, помеченные как аномальные\")\n            cols = ([\"Timestamp\"] if \"Timestamp\" in analyzed.columns else []) +\\\n                [\"Destination Port\", \"Flow Duration\", \"Total Fwd Packets\", \"Flow Packets/s\",\n                 \"if_score\"] + [c for c in [\"rf_prediction\", \"rf_confidence\",\n                                            config.NETWORK_LABEL_COL] if c in analyzed.columns]\n            anomalies = analyzed[analyzed[\"if_anomaly\"]].sort_values(\"if_score\", ascending=False)\n            st.dataframe(anomalies[cols].head(300), width=\"stretch\", hide_index=True)\n            st.download_button(\"⬇️ Полный результат анализа (CSV)\", to_csv_bytes(analyzed),\n                               \"network_analysis.csv\", \"text/csv\")\n\nwith tab_http:\n    from src.nlp.models import classify_requests, explain_request, http_events\n    from src.nlp.preprocess import load_text_csv, normalize_request\n\n    st.subheader(\"Классификация HTTP-запросов: Normal / SQL Injection / XSS\")\n    if nlp_model is None:\n        st.warning(\"NLP-модель не найдена. Подготовьте датасет (`scripts/prepare_text_dataset.py`) \"\n                   \"и выполните `python train_all.py --only nlp`.\")\n    else:\n        text = st.text_area(\"Проверить один запрос (строка запроса, параметр или тело формы)\",\n                            height=90, max_chars=config.MAX_REQUEST_LENGTH)\n        if st.button(\"Классифицировать\") and text.strip():\n            res = classify_requests(nlp_model, [text]).iloc[0]\n            c1, c2 = st.columns([1, 2])\n            c1.metric(\"Прогноз модели\", res[\"prediction\"], f\"уверенность {res['confidence']:.2f}\")\n            if res[\"needs_review\"]:\n                c1.warning(\"Низкая уверенность — нужна ручная проверка.\")\n            probs = {c: res[f\"prob_{c}\"] for c in nlp_model.classes_}\n            c2.bar_chart(pd.Series(probs, name=\"вероятность\"))\n            st.markdown(\"Нормализованный запрос (после декодирования):\")\n            st.code(normalize_request(text), language=None)\n            st.markdown(\"Наиболее повлиявшие символьные n-граммы:\")\n            st.dataframe(explain_request(nlp_model, text), hide_index=True)\n\n        st.divider()\n        up = st.file_uploader(\"Пакетная проверка: CSV со столбцом `request` \"\n                              \"(необязательно `label`, `timestamp`)\", type=[\"csv\"], key=\"http_upload\")\n        if up is not None and st.button(\"Проанализировать запросы\"):\n            try:\n                df = load_text_csv(up, filename=up.name)\n                res = classify_requests(nlp_model, df[config.TEXT_COL])\n                set_events(\"http\", http_events(df, res))\n                st.session_state[\"http_batch\"] = pd.concat(\n                    [res, df.drop(columns=[config.TEXT_COL])], axis=1)\n                st.success(f\"Классифицировано {len(res)} запросов. Результаты добавлены на «Обзор».\")\n            except (DataValidationError, ValueError) as exc:\n                st.error(f\"Файл отклонён: {exc}\")\n\n        batch = st.session_state.get(\"http_batch\")\n        if batch is not None:\n            st.dataframe(batch.head(500), width=\"stretch\", hide_index=True)\n            st.download_button(\"⬇️ Результаты классификации (CSV)\", to_csv_bytes(batch),\n                               \"http_classification.csv\", \"text/csv\")\n\nwith tab_eval:\n    st.subheader(\"Метрики на тестовой выборке\")\n    if not config.METRICS_PATH.exists():\n        st.info(\"Метрик нет — выполните `python train_all.py`.\")\n    else:\n        metrics = json.loads(config.METRICS_PATH.read_text(encoding=\"utf-8\"))\n        rows = []\n        for module, models in [(\"network\", [\"isolation_forest\", \"random_forest\"]),\n                               (\"nlp\", [\"logistic_regression\", \"naive_bayes\"])]:\n            for m in models:\n                mm = metrics.get(module, {}).get(m)\n                if mm:\n                    rows.append({\"Модуль\": module, \"Модель\": m, \"Accuracy\": mm[\"accuracy\"],\n                                 \"Precision (macro)\": mm[\"precision_macro\"],\n                                 \"Recall (macro)\": mm[\"recall_macro\"], \"F1 (macro)\": mm[\"f1_macro\"],\n                                 \"FPR\": mm.get(\"false_positive_rate\"), \"Test size\": mm[\"support\"]})\n        st.dataframe(pd.DataFrame(rows), hide_index=True, width=\"stretch\")\n        st.caption(\"FPR — доля нормальных объектов, ошибочно помеченных как атака/аномалия. \"\n                   \"При дисбалансе классов ориентируйтесь на Recall/F1 по классам атак, а не на Accuracy.\")\n\n        imgs = sorted(config.REPORTS_DIR.glob(\"cm_*.png\"))\n        for i in range(0, len(imgs), 2):\n            cols = st.columns(2)\n            for col, img in zip(cols, imgs[i:i + 2]):\n                col.image(str(img), caption=img.stem)\n\n        iso = metrics.get(\"network\", {}).get(\"isolation_forest\")\n        if iso:\n            st.markdown(\"##### Isolation Forest: доля помеченных записей по фактической метке\")\n            st.bar_chart(pd.Series(iso[\"detection_rate_by_label\"], name=\"доля помеченных\"))\n            st.caption(\"Для BENIGN это доля ложных срабатываний. Типы атак, похожие на норму \"\n                       \"по статистике потока, детектор аномалий пропускает — аномальность \"\n                       \"и вредоносность не одно и то же.\")\n        with st.expander(\"Полный metrics.json\"):\n            st.json(metrics)\n\nwith tab_about:\n    st.markdown(f\"\"\"\n**Что делает система.** Анализирует подготовленные наборы данных: сетевые потоки\n(Isolation Forest + Random Forest) и тексты HTTP-запросов (TF-IDF + Logistic Regression)\nи формирует предупреждения. Система ничего не блокирует и не атакует.\n\n**Ограничения.**\n* Прогноз модели — не подтверждённый инцидент. Возможны ложные срабатывания и пропуски атак.\n* Модели обучены на учебных данных; на реальном трафике качество будет ниже.\n* Новые техники обфускации и типы атак, которых не было в обучении, могут не распознаваться.\n* Порог ручной проверки NLP-модели: уверенность < {config.NLP_REVIEW_THRESHOLD}.\n\n**Безопасность.** Принимаются только CSV до {config.MAX_UPLOAD_MB} МБ и\n{config.MAX_UPLOAD_ROWS:,} строк; содержимое не исполняется; при выгрузке ячейки,\nначинающиеся с `= + - @`, экранируются.\n\"\"\")\n",
  ".streamlit/config.toml": "[server]\nmaxUploadSize = 20\naddress = \"0.0.0.0\"\nport = 8501\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\nenableWebsocketCompression = false\n[browser]\ngatherUsageStats = false\n",
  "src/__init__.py": "",
  "src/common/__init__.py": "",
  "src/common/io_utils.py": "from __future__ import annotations\n\nimport io\nfrom pathlib import Path\nfrom typing import Iterable\n\nimport numpy as np\nimport pandas as pd\n\nimport config\n\nclass DataValidationError(ValueError):\n    pass\n\ndef read_csv_safely(\n    source,\n    max_mb: float = config.MAX_UPLOAD_MB,\n    max_rows: int = config.MAX_UPLOAD_ROWS,\n    filename: str | None = None,\n) -> pd.DataFrame:\n    if isinstance(source, (str, Path)):\n        path = Path(source)\n        name = filename or path.name\n        size_mb = path.stat().st_size / 1024 / 1024\n        raw = None\n    else:\n        raw = source if isinstance(source, bytes) else source.read()\n        name = filename or getattr(source, \"name\", \"upload.csv\")\n        size_mb = len(raw) / 1024 / 1024\n\n    if Path(name).suffix.lower() not in config.ALLOWED_EXTENSIONS:\n        raise DataValidationError(f\"Допустимы только CSV-файлы, получен: {name}\")\n    if size_mb > max_mb:\n        raise DataValidationError(\n            f\"Файл {size_mb:.1f} МБ превышает лимит {max_mb} МБ\"\n        )\n\n    try:\n        if raw is None:\n            df = pd.read_csv(path, nrows=max_rows + 1, low_memory=False,\n                             encoding_errors=\"replace\")\n        else:\n            df = pd.read_csv(io.BytesIO(raw), nrows=max_rows + 1,\n                             low_memory=False, encoding_errors=\"replace\")\n    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:\n        raise DataValidationError(f\"Не удалось разобрать CSV: {exc}\") from exc\n\n    if len(df) > max_rows:\n        raise DataValidationError(f\"Слишком много строк (> {max_rows})\")\n    if df.empty:\n        raise DataValidationError(\"Файл не содержит данных\")\n\n    df.columns = [str(c).strip() for c in df.columns]\n    return df\n\ndef require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:\n    missing = [c for c in required if c not in df.columns]\n    if missing:\n        raise DataValidationError(\n            \"В файле нет обязательных столбцов: \" + \", \".join(missing)\n        )\n\ndef data_quality_report(df: pd.DataFrame) -> dict:\n    numeric = df.select_dtypes(include=[np.number])\n    return {\n        \"rows\": int(len(df)),\n        \"columns\": int(df.shape[1]),\n        \"duplicates\": int(df.duplicated().sum()),\n        \"missing_total\": int(df.isna().sum().sum()),\n        \"missing_by_column\": {k: int(v) for k, v in df.isna().sum().items() if v > 0},\n        \"infinite_values\": int(np.isinf(numeric.to_numpy()).sum()) if not numeric.empty else 0,\n    }\n\n_FORMULA_PREFIXES = (\"=\", \"+\", \"-\", \"@\", \"\\t\", \"\\r\")\n\ndef sanitize_for_csv_export(df: pd.DataFrame) -> pd.DataFrame:\n    out = df.copy()\n    for col in out.columns:\n\n        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):\n            out[col] = out[col].map(\n                lambda v: \"'\" + v if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v\n            )\n    return out\n\ndef to_csv_bytes(df: pd.DataFrame) -> bytes:\n    return sanitize_for_csv_export(df).to_csv(index=False).encode(\"utf-8-sig\")\n\ndef make_events_frame(**columns) -> pd.DataFrame:\n    missing = [c for c in config.EVENT_COLUMNS if c not in columns]\n    if missing:\n        raise ValueError(f\"Не заполнены поля события: {missing}\")\n    df = pd.DataFrame({c: columns[c] for c in config.EVENT_COLUMNS})\n    df[\"score\"] = df[\"score\"].astype(float).round(4)\n    df[\"is_alert\"] = df[\"is_alert\"].astype(bool)\n    df[\"needs_review\"] = df[\"needs_review\"].astype(bool)\n    return df\n\ndef validate_events(df: pd.DataFrame) -> None:\n    require_columns(df, config.EVENT_COLUMNS)\n    bad_modules = set(df[\"source_module\"].unique()) - {\"network\", \"http\"}\n    if bad_modules:\n        raise DataValidationError(f\"Неизвестные source_module: {bad_modules}\")\n    if not df[\"score\"].between(0, 1).all():\n        raise DataValidationError(\"score должен быть в диапазоне [0, 1]\")\n\ndef save_events(df: pd.DataFrame, path: Path) -> None:\n    validate_events(df)\n    path.parent.mkdir(parents=True, exist_ok=True)\n    df.to_csv(path, index=False)\n\ndef load_events(path: Path) -> pd.DataFrame:\n    if not Path(path).exists():\n        return pd.DataFrame(columns=config.EVENT_COLUMNS)\n    df = pd.read_csv(path, dtype={\"ground_truth\": str, \"timestamp\": str})\n    df[\"is_alert\"] = df[\"is_alert\"].astype(bool)\n    df[\"needs_review\"] = df[\"needs_review\"].astype(bool)\n    return df\n",
  "src/common/metrics.py": "from __future__ import annotations\n\nimport matplotlib\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport seaborn as sns\nfrom sklearn.metrics import (\n    accuracy_score,\n    classification_report,\n    confusion_matrix,\n    precision_recall_fscore_support,\n)\n\ndef evaluate_classifier(y_true, y_pred, labels: list[str]) -> dict:\n    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(\n        y_true, y_pred, labels=labels, average=\"macro\", zero_division=0\n    )\n    p_w, r_w, f_w, _ = precision_recall_fscore_support(\n        y_true, y_pred, labels=labels, average=\"weighted\", zero_division=0\n    )\n    report = classification_report(\n        y_true, y_pred, labels=labels, output_dict=True, zero_division=0\n    )\n    cm = confusion_matrix(y_true, y_pred, labels=labels)\n    return {\n        \"accuracy\": round(float(accuracy_score(y_true, y_pred)), 4),\n        \"precision_macro\": round(float(p_macro), 4),\n        \"recall_macro\": round(float(r_macro), 4),\n        \"f1_macro\": round(float(f_macro), 4),\n        \"precision_weighted\": round(float(p_w), 4),\n        \"recall_weighted\": round(float(r_w), 4),\n        \"f1_weighted\": round(float(f_w), 4),\n        \"per_class\": {\n            lbl: {k: round(float(v), 4) for k, v in report[lbl].items()}\n            for lbl in labels if lbl in report\n        },\n        \"labels\": list(labels),\n        \"confusion_matrix\": cm.tolist(),\n        \"support\": int(len(y_true)),\n    }\n\ndef false_positive_rate(y_true_binary, y_pred_binary) -> float:\n    y_true_binary = np.asarray(y_true_binary, dtype=bool)\n    y_pred_binary = np.asarray(y_pred_binary, dtype=bool)\n    fp = np.sum(~y_true_binary & y_pred_binary)\n    tn = np.sum(~y_true_binary & ~y_pred_binary)\n    return float(fp / (fp + tn)) if (fp + tn) else 0.0\n\ndef plot_confusion_matrix(cm, labels, title: str, path) -> None:\n    cm = np.asarray(cm)\n    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2.5, 1.2 * len(labels) + 2))\n    sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\", cbar=False,\n                xticklabels=labels, yticklabels=labels, ax=ax)\n    ax.set_xlabel(\"Прогноз модели\")\n    ax.set_ylabel(\"Истинная метка\")\n    ax.set_title(title)\n    plt.xticks(rotation=30, ha=\"right\")\n    plt.yticks(rotation=0)\n    fig.tight_layout()\n    fig.savefig(path, dpi=120)\n    plt.close(fig)\n",
  "src/network/__init__.py": "",
  "src/network/preprocess.py": "from __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\nimport config\nfrom src.common.io_utils import data_quality_report, read_csv_safely, require_columns\n\ndef load_network_csv(source, filename: str | None = None, require_label: bool = False) -> pd.DataFrame:\n    df = read_csv_safely(source, filename=filename)\n    required = list(config.NETWORK_FEATURES)\n    if require_label:\n        required.append(config.NETWORK_LABEL_COL)\n    require_columns(df, required)\n    return df\n\ndef clean_network_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:\n    report = data_quality_report(df)\n    out = df.copy()\n\n    for col in config.NETWORK_FEATURES:\n\n        out[col] = pd.to_numeric(out[col], errors=\"coerce\")\n    out[config.NETWORK_FEATURES] = out[config.NETWORK_FEATURES].replace([np.inf, -np.inf], np.nan)\n\n    out = out.dropna(subset=config.NETWORK_FEATURES, how=\"all\")\n\n    if config.NETWORK_LABEL_COL in out.columns:\n        out[config.NETWORK_LABEL_COL] = out[config.NETWORK_LABEL_COL].astype(str).str.strip()\n\n        out[config.NETWORK_LABEL_COL] = out[config.NETWORK_LABEL_COL].str.replace(\"�\", \"-\", regex=False)\n\n    before = len(out)\n    out = out.drop_duplicates().reset_index(drop=True)\n    report[\"duplicates_removed\"] = int(before - len(out))\n    report[\"rows_after_cleaning\"] = int(len(out))\n    return out, report\n\ndef to_binary_attack(labels: pd.Series) -> np.ndarray:\n    return (labels.astype(str) != config.NETWORK_BENIGN_LABEL).to_numpy().astype(int)\n\ndef parse_timestamps(df: pd.DataFrame) -> pd.Series:\n    if config.NETWORK_TIME_COL not in df.columns:\n        return pd.Series([\"\"] * len(df), index=df.index)\n    ts = pd.to_datetime(df[config.NETWORK_TIME_COL], errors=\"coerce\", format=\"mixed\")\n    return ts.dt.strftime(\"%Y-%m-%dT%H:%M:%S\").fillna(\"\")\n",
  "src/network/models.py": "from __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import IsolationForest, RandomForestClassifier\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.metrics import f1_score\nfrom sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n\nimport config\nfrom src.common.io_utils import make_events_frame\nfrom src.common.metrics import evaluate_classifier, false_positive_rate\nfrom src.network.preprocess import parse_timestamps, to_binary_attack\n\nFEATURES = config.NETWORK_FEATURES\n\ndef build_isoforest_pipeline(contamination: float = config.ISOFOREST_CONTAMINATION) -> Pipeline:\n    return Pipeline([\n        (\"imputer\", SimpleImputer(strategy=\"median\")),\n        (\"scaler\", StandardScaler()),\n        (\"model\", IsolationForest(\n            n_estimators=200,\n            contamination=contamination,\n            random_state=config.RANDOM_STATE,\n            n_jobs=-1,\n        )),\n    ])\n\ndef build_rf_pipeline() -> Pipeline:\n    return Pipeline([\n        (\"imputer\", SimpleImputer(strategy=\"median\")),\n        (\"model\", RandomForestClassifier(\n            n_estimators=200,\n            class_weight=\"balanced_subsample\",\n            random_state=config.RANDOM_STATE,\n            n_jobs=-1,\n        )),\n    ])\n\ndef tune_isoforest_contamination(X_train: pd.DataFrame, y_train_bin: np.ndarray,\n                                 grid=(0.01, 0.02, 0.05, 0.10, 0.15)) -> float:\n    if y_train_bin.sum() == 0 or y_train_bin.sum() == len(y_train_bin):\n        return config.ISOFOREST_CONTAMINATION\n    X_fit, X_val, y_fit, y_val = train_test_split(\n        X_train, y_train_bin, test_size=0.25, stratify=y_train_bin,\n        random_state=config.RANDOM_STATE,\n    )\n    best_c, best_f1 = config.ISOFOREST_CONTAMINATION, -1.0\n    for c in grid:\n        pipe = build_isoforest_pipeline(c).fit(X_fit[y_fit == 0])\n        pred = (pipe.predict(X_val) == -1).astype(int)\n        f1 = f1_score(y_val, pred, zero_division=0)\n        if f1 > best_f1:\n            best_c, best_f1 = c, f1\n    return best_c\n\ndef train_network_models(df: pd.DataFrame) -> tuple[dict, dict]:\n    label_col = config.NETWORK_LABEL_COL\n    data = df.copy()\n\n    counts = data[label_col].value_counts()\n    rare = counts[counts < 10].index\n    if len(rare):\n        data.loc[data[label_col].isin(rare), label_col] = \"Other attack\"\n\n    X = data[FEATURES]\n    y = data[label_col].astype(str)\n    X_train, X_test, y_train, y_test = train_test_split(\n        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE\n    )\n    y_train_bin = to_binary_attack(y_train)\n    y_test_bin = to_binary_attack(y_test)\n\n    contamination = tune_isoforest_contamination(X_train, y_train_bin)\n    iso = build_isoforest_pipeline(contamination).fit(X_train[y_train_bin == 0])\n    iso_pred = (iso.predict(X_test) == -1).astype(int)\n    iso_metrics = evaluate_classifier(\n        np.where(y_test_bin == 1, \"Attack\", \"Normal\"),\n        np.where(iso_pred == 1, \"Attack\", \"Normal\"),\n        labels=[\"Normal\", \"Attack\"],\n    )\n    iso_metrics[\"false_positive_rate\"] = round(false_positive_rate(y_test_bin, iso_pred), 4)\n    iso_metrics[\"contamination\"] = contamination\n\n    iso_metrics[\"detection_rate_by_label\"] = {\n        lbl: round(float(iso_pred[(y_test == lbl).to_numpy()].mean()), 4)\n        for lbl in sorted(y_test.unique())\n    }\n\n    grid = GridSearchCV(\n        build_rf_pipeline(),\n        param_grid={\"model__max_depth\": [None, 20], \"model__min_samples_leaf\": [1, 3]},\n        scoring=\"f1_macro\",\n        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),\n        n_jobs=-1,\n    )\n    grid.fit(X_train, y_train)\n    rf = grid.best_estimator_\n    rf_labels = sorted(y.unique(), key=lambda s: (s != config.NETWORK_BENIGN_LABEL, s))\n    rf_metrics = evaluate_classifier(y_test, rf.predict(X_test), labels=rf_labels)\n    rf_metrics[\"best_params\"] = grid.best_params_\n    rf_metrics[\"cv_f1_macro\"] = round(float(grid.best_score_), 4)\n    rf_metrics[\"false_positive_rate\"] = round(\n        false_positive_rate(y_test_bin, to_binary_attack(pd.Series(rf.predict(X_test)))), 4\n    )\n    importances = rf.named_steps[\"model\"].feature_importances_\n    rf_metrics[\"feature_importance\"] = {\n        f: round(float(v), 4)\n        for f, v in sorted(zip(FEATURES, importances), key=lambda t: -t[1])\n    }\n\n    models = {\"isoforest\": iso, \"random_forest\": rf}\n    metrics = {\n        \"dataset\": {\n            \"train_rows\": int(len(X_train)),\n            \"test_rows\": int(len(X_test)),\n            \"class_distribution\": {k: int(v) for k, v in y.value_counts().items()},\n        },\n        \"isolation_forest\": iso_metrics,\n        \"random_forest\": rf_metrics,\n    }\n    return models, metrics\n\ndef analyze_network(df: pd.DataFrame, iso: Pipeline, rf: Pipeline | None = None) -> pd.DataFrame:\n    X = df[FEATURES]\n    out = df.copy()\n    out[\"if_anomaly\"] = iso.predict(X) == -1\n\n    out[\"if_score\"] = np.clip(-iso.score_samples(X), 0, 1)\n    if rf is not None:\n        proba = rf.predict_proba(X)\n        classes = rf.classes_\n        out[\"rf_prediction\"] = classes[proba.argmax(axis=1)]\n        out[\"rf_confidence\"] = proba.max(axis=1)\n    return out\n\ndef network_events(analyzed: pd.DataFrame) -> pd.DataFrame:\n    n = len(analyzed)\n    has_rf = \"rf_prediction\" in analyzed.columns\n    if has_rf:\n        rf_attack = analyzed[\"rf_prediction\"] != config.NETWORK_BENIGN_LABEL\n        needs_review = analyzed[\"if_anomaly\"] != rf_attack\n        rf_text = (\"RF: \" + analyzed[\"rf_prediction\"].astype(str)\n                   + \" (\" + analyzed[\"rf_confidence\"].round(2).astype(str) + \")\")\n    else:\n        needs_review = pd.Series(False, index=analyzed.index)\n        rf_text = pd.Series(\"RF: —\", index=analyzed.index)\n\n    details = (\n        \"port \" + analyzed[\"Destination Port\"].fillna(-1).astype(int).astype(str)\n        + \" | fwd_pkts \" + analyzed[\"Total Fwd Packets\"].fillna(0).astype(int).astype(str)\n        + \" | \" + rf_text\n    )\n    truth = (analyzed[config.NETWORK_LABEL_COL].astype(str)\n             if config.NETWORK_LABEL_COL in analyzed.columns else [\"\"] * n)\n\n    return make_events_frame(\n        event_id=[f\"net-{i:06d}\" for i in range(1, n + 1)],\n        timestamp=parse_timestamps(analyzed).to_numpy(),\n        source_module=[\"network\"] * n,\n        model_name=[\"IsolationForest\"] * n,\n        prediction=np.where(analyzed[\"if_anomaly\"], \"Anomaly\", \"Normal\"),\n        score=analyzed[\"if_score\"].to_numpy(),\n        is_alert=analyzed[\"if_anomaly\"].to_numpy(),\n        needs_review=needs_review.to_numpy(),\n        ground_truth=np.asarray(truth),\n        details=details.to_numpy(),\n        status=[config.PREDICTION_STATUS] * n,\n    )\n",
  "src/nlp/__init__.py": "",
  "src/nlp/preprocess.py": "from __future__ import annotations\n\nimport html\nimport re\nfrom urllib.parse import unquote_plus\n\nimport pandas as pd\n\nimport config\nfrom src.common.io_utils import read_csv_safely, require_columns\n\n_WS_RE = re.compile(r\"\\s+\")\n_DIGITS_RE = re.compile(r\"\\d+\")\n_LABEL_ALIASES = {\n    \"normal\": \"Normal\", \"benign\": \"Normal\", \"0\": \"Normal\",\n    \"sql injection\": \"SQL Injection\", \"sqli\": \"SQL Injection\", \"sql\": \"SQL Injection\",\n    \"xss\": \"XSS\", \"cross-site scripting\": \"XSS\",\n}\n\ndef normalize_request(text) -> str:\n    if text is None or (isinstance(text, float) and pd.isna(text)):\n        return \"\"\n    s = str(text)[: config.MAX_REQUEST_LENGTH]\n    for _ in range(3):\n        decoded = unquote_plus(s)\n        if decoded == s:\n            break\n        s = decoded\n    s = html.unescape(s)\n    s = s.replace(\"\\x00\", \" \").lower()\n    s = _DIGITS_RE.sub(\"0\", s)\n    s = _WS_RE.sub(\" \", s).strip()\n    return s\n\ndef normalize_label(value) -> str:\n    key = str(value).strip().lower()\n    return _LABEL_ALIASES.get(key, str(value).strip())\n\ndef load_text_csv(source, filename: str | None = None, require_label: bool = False) -> pd.DataFrame:\n    df = read_csv_safely(source, filename=filename)\n    required = [config.TEXT_COL] + ([config.TEXT_LABEL_COL] if require_label else [])\n    require_columns(df, required)\n    df[config.TEXT_COL] = df[config.TEXT_COL].fillna(\"\").astype(str)\n    if config.TEXT_LABEL_COL in df.columns:\n        df[config.TEXT_LABEL_COL] = df[config.TEXT_LABEL_COL].map(normalize_label)\n        unknown = set(df[config.TEXT_LABEL_COL].unique()) - set(config.TEXT_CLASSES)\n        if unknown and require_label:\n            raise ValueError(f\"Неизвестные метки классов: {sorted(unknown)}\")\n    return df\n\ndef clean_text_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:\n    out = df.copy()\n    out[\"_norm\"] = out[config.TEXT_COL].map(normalize_request)\n    report = {\"rows\": int(len(out)), \"empty_removed\": int((out[\"_norm\"] == \"\").sum())}\n    out = out[out[\"_norm\"] != \"\"]\n    before = len(out)\n    out = out.drop_duplicates(subset=\"_norm\").reset_index(drop=True)\n    report[\"duplicates_removed\"] = int(before - len(out))\n    report[\"rows_after_cleaning\"] = int(len(out))\n    if config.TEXT_LABEL_COL in out.columns:\n        report[\"class_distribution\"] = {\n            k: int(v) for k, v in out[config.TEXT_LABEL_COL].value_counts().items()\n        }\n    return out.drop(columns=\"_norm\"), report\n",
  "src/nlp/models.py": "from __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.pipeline import Pipeline\n\nimport config\nfrom src.common.io_utils import make_events_frame\nfrom src.common.metrics import evaluate_classifier, false_positive_rate\nfrom src.nlp.preprocess import normalize_request\n\nCLASSES = config.TEXT_CLASSES\n\ndef build_vectorizer() -> TfidfVectorizer:\n    return TfidfVectorizer(\n        preprocessor=normalize_request,\n        lowercase=False,\n        analyzer=\"char_wb\",\n        ngram_range=(2, 5),\n        min_df=2,\n        max_features=50_000,\n        sublinear_tf=True,\n    )\n\ndef build_logreg_pipeline() -> Pipeline:\n    return Pipeline([\n        (\"tfidf\", build_vectorizer()),\n        (\"model\", LogisticRegression(max_iter=3000, class_weight=\"balanced\")),\n    ])\n\ndef build_nb_pipeline() -> Pipeline:\n    return Pipeline([\n        (\"tfidf\", build_vectorizer()),\n        (\"model\", MultinomialNB()),\n    ])\n\ndef train_nlp_models(df: pd.DataFrame) -> tuple[dict, dict]:\n    X = df[config.TEXT_COL].astype(str)\n    y = df[config.TEXT_LABEL_COL].astype(str)\n    X_train, X_test, y_train, y_test = train_test_split(\n        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE\n    )\n    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)\n\n    candidates = {\n        \"logistic_regression\": (build_logreg_pipeline(), {\"model__C\": [1, 10, 30]}),\n        \"naive_bayes\": (build_nb_pipeline(), {\"model__alpha\": [0.01, 0.1, 1.0]}),\n    }\n    models, metrics = {}, {}\n    for name, (pipe, grid) in candidates.items():\n        search = GridSearchCV(pipe, grid, scoring=\"f1_macro\", cv=cv, n_jobs=-1)\n        search.fit(X_train, y_train)\n        best = search.best_estimator_\n        y_pred = best.predict(X_test)\n        m = evaluate_classifier(y_test, y_pred, labels=CLASSES)\n        m[\"best_params\"] = search.best_params_\n        m[\"cv_f1_macro\"] = round(float(search.best_score_), 4)\n\n        m[\"false_positive_rate\"] = round(\n            false_positive_rate(y_test != \"Normal\", y_pred != \"Normal\"), 4\n        )\n        models[name], metrics[name] = best, m\n\n    metrics[\"dataset\"] = {\n        \"train_rows\": int(len(X_train)),\n        \"test_rows\": int(len(X_test)),\n        \"class_distribution\": {k: int(v) for k, v in y.value_counts().items()},\n    }\n    return models, metrics\n\ndef classify_requests(model: Pipeline, requests) -> pd.DataFrame:\n    requests = pd.Series(list(requests), dtype=str)\n    proba = model.predict_proba(requests)\n    classes = list(model.classes_)\n    out = pd.DataFrame({\"request\": requests})\n    out[\"prediction\"] = np.array(classes)[proba.argmax(axis=1)]\n    out[\"confidence\"] = proba.max(axis=1).round(4)\n    for i, c in enumerate(classes):\n        out[f\"prob_{c}\"] = proba[:, i].round(4)\n    out[\"needs_review\"] = out[\"confidence\"] < config.NLP_REVIEW_THRESHOLD\n    return out\n\ndef explain_request(model: Pipeline, request: str, top_k: int = 8) -> pd.DataFrame:\n    clf = model.named_steps[\"model\"]\n    if not hasattr(clf, \"coef_\"):\n        return pd.DataFrame(columns=[\"ngram\", \"contribution\"])\n    vec = model.named_steps[\"tfidf\"]\n    x = vec.transform([request])\n    cls_idx = int(np.argmax(clf.predict_proba(x)[0]))\n    coef = clf.coef_[cls_idx]\n    idx = x.nonzero()[1]\n    contrib = x.toarray()[0, idx] * coef[idx]\n    names = vec.get_feature_names_out()[idx]\n    df = pd.DataFrame({\"ngram\": names, \"contribution\": contrib.round(4)})\n    return df.sort_values(\"contribution\", ascending=False).head(top_k).reset_index(drop=True)\n\ndef http_events(source_df: pd.DataFrame, classified: pd.DataFrame) -> pd.DataFrame:\n    n = len(classified)\n    ts = (pd.to_datetime(source_df[config.TEXT_TIME_COL], errors=\"coerce\", format=\"mixed\")\n          .dt.strftime(\"%Y-%m-%dT%H:%M:%S\").fillna(\"\").to_numpy()\n          if config.TEXT_TIME_COL in source_df.columns else [\"\"] * n)\n    truth = (source_df[config.TEXT_LABEL_COL].astype(str).to_numpy()\n             if config.TEXT_LABEL_COL in source_df.columns else [\"\"] * n)\n    return make_events_frame(\n        event_id=[f\"http-{i:06d}\" for i in range(1, n + 1)],\n        timestamp=ts,\n        source_module=[\"http\"] * n,\n        model_name=[\"LogisticRegression\"] * n,\n        prediction=classified[\"prediction\"].to_numpy(),\n        score=classified[\"confidence\"].to_numpy(),\n        is_alert=(classified[\"prediction\"] != \"Normal\").to_numpy(),\n        needs_review=classified[\"needs_review\"].to_numpy(),\n        ground_truth=truth,\n        details=classified[\"request\"].str.slice(0, 160).to_numpy(),\n        status=[config.PREDICTION_STATUS] * n,\n    )\n",
  "scripts/generate_network_data.py": "from __future__ import annotations\n\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nimport config\n\nCOMMON_PORTS = [80, 443, 443, 443, 53, 53, 22, 8080, 25, 123, 3389, 993]\n\ndef _flows(rng: np.random.Generator, n: int, kind: str) -> pd.DataFrame:\n    if kind == \"BENIGN\":\n        port = rng.choice(COMMON_PORTS, n)\n        duration = rng.lognormal(12, 2.2, n)\n        fwd = rng.poisson(9, n) + 1\n        bwd = rng.poisson(8, n)\n        fwd_mean = rng.lognormal(4.2, 1.0, n)\n        bwd_mean = rng.lognormal(5.5, 1.3, n)\n        syn = rng.binomial(1, 0.05, n)\n\n        heavy = rng.random(n) < 0.04\n        fwd = np.where(heavy, fwd * 40, fwd)\n        bwd = np.where(heavy, bwd * 60, bwd)\n        duration = np.where(heavy, duration * 30, duration)\n    elif kind == \"DoS Hulk\":\n        port = np.full(n, 80)\n        duration = rng.lognormal(10.5, 1.5, n)\n        fwd = rng.poisson(6, n) + 2\n        bwd = rng.poisson(1, n)\n        fwd_mean = rng.normal(55, 25, n).clip(0)\n        bwd_mean = rng.normal(1500, 900, n).clip(0)\n        syn = rng.binomial(1, 0.2, n)\n    elif kind == \"DDoS\":\n        port = np.full(n, 80)\n        duration = rng.lognormal(9.5, 1.8, n)\n        fwd = rng.poisson(3, n) + 1\n        bwd = rng.poisson(4, n)\n        fwd_mean = rng.normal(20, 10, n).clip(0)\n        bwd_mean = rng.normal(1800, 900, n).clip(0)\n        syn = rng.binomial(1, 0.1, n)\n    elif kind == \"PortScan\":\n        port = rng.integers(1, 65535, n)\n        duration = rng.lognormal(4, 1.5, n)\n        fwd = rng.integers(1, 3, n)\n        bwd = rng.integers(0, 2, n)\n        fwd_mean = rng.normal(2, 3, n).clip(0)\n        bwd_mean = rng.normal(3, 4, n).clip(0)\n        syn = np.ones(n, dtype=int)\n    elif kind == \"SSH-Patator\":\n        port = np.full(n, 22)\n        duration = rng.lognormal(14, 0.6, n)\n        fwd = rng.poisson(22, n) + 5\n        bwd = rng.poisson(33, n) + 5\n        fwd_mean = rng.normal(90, 20, n).clip(0)\n        bwd_mean = rng.normal(100, 25, n).clip(0)\n        syn = rng.binomial(1, 0.3, n)\n    else:\n        raise ValueError(kind)\n\n    fwd_len = fwd * fwd_mean\n    bwd_len = bwd * bwd_mean\n    total_pkts = fwd + bwd\n    seconds = duration / 1e6\n    with np.errstate(divide=\"ignore\", invalid=\"ignore\"):\n        bytes_s = (fwd_len + bwd_len) / seconds\n        pkts_s = total_pkts / seconds\n        iat = duration / np.maximum(total_pkts - 1, 1)\n        avg = (fwd_len + bwd_len) / np.maximum(total_pkts, 1)\n\n    return pd.DataFrame({\n        \"Destination Port\": port,\n        \"Flow Duration\": duration.round(0),\n        \"Total Fwd Packets\": fwd,\n        \"Total Backward Packets\": bwd,\n        \"Total Length of Fwd Packets\": fwd_len.round(1),\n        \"Total Length of Bwd Packets\": bwd_len.round(1),\n        \"Fwd Packet Length Mean\": fwd_mean.round(2),\n        \"Bwd Packet Length Mean\": bwd_mean.round(2),\n        \"Flow Bytes/s\": bytes_s.round(2),\n        \"Flow Packets/s\": pkts_s.round(2),\n        \"Flow IAT Mean\": iat.round(2),\n        \"SYN Flag Count\": syn,\n        \"ACK Flag Count\": rng.binomial(1, 0.6, n),\n        \"Average Packet Size\": avg.round(2),\n        \"Label\": kind,\n    })\n\ndef make_network(n_total: int, seed: int, start: str) -> pd.DataFrame:\n    rng = np.random.default_rng(seed)\n    shares = {\"BENIGN\": 0.74, \"DoS Hulk\": 0.09, \"PortScan\": 0.08,\n              \"DDoS\": 0.06, \"SSH-Patator\": 0.03}\n    parts = [_flows(rng, int(n_total * s), k) for k, s in shares.items()]\n    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)\n\n    idx = rng.choice(len(df), size=int(0.01 * len(df)), replace=False)\n    df.iloc[idx, df.columns.get_loc(\"Flow Bytes/s\")] = np.inf\n    idx = rng.choice(len(df), size=int(0.01 * len(df)), replace=False)\n    df.iloc[idx, df.columns.get_loc(\"Flow IAT Mean\")] = np.nan\n    ts = pd.Timestamp(start) + pd.to_timedelta(\n        np.sort(rng.uniform(0, 8 * 3600, len(df))), unit=\"s\")\n    df.insert(0, \"Timestamp\", ts.strftime(\"%Y-%m-%d %H:%M:%S\"))\n\n    df = pd.concat([df, df.sample(frac=0.02, random_state=seed)], ignore_index=True)\n    return df.sort_values(\"Timestamp\").reset_index(drop=True)\n\ndef main() -> None:\n    config.DATA_DIR.mkdir(parents=True, exist_ok=True)\n    make_network(12_000, seed=1, start=\"2026-09-01 08:00\").to_csv(config.NETWORK_DATA_PATH, index=False)\n    make_network(2_000, seed=7, start=\"2026-09-02 08:00\").to_csv(\n        config.DATA_DIR / \"network_traffic_new.csv\", index=False)\n    print(\"Сетевые данные записаны в\", config.DATA_DIR)\n\nif __name__ == \"__main__\":\n    main()\n",
  "scripts/prepare_text_dataset.py": "from __future__ import annotations\n\nimport argparse\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nimport config\nfrom src.common.io_utils import read_csv_safely, require_columns\n\nATTACK_VALUES = {\"1\", \"1.0\", \"true\", \"attack\", \"malicious\"}\n\ndef _load_binary(path: str, text_col: str, label_col: str, attack_class: str) -> pd.DataFrame:\n    df = read_csv_safely(path, max_mb=500, max_rows=2_000_000)\n    require_columns(df, [text_col, label_col])\n    out = pd.DataFrame({\n        config.TEXT_COL: df[text_col].astype(str).str.strip(),\n        config.TEXT_LABEL_COL: np.where(\n            df[label_col].astype(str).str.strip().str.lower().isin(ATTACK_VALUES),\n            attack_class, \"Normal\"),\n    })\n    return out[out[config.TEXT_COL] != \"\"]\n\ndef build_dataset(args) -> pd.DataFrame:\n    parts = []\n    if args.sqli:\n        parts.append(_load_binary(args.sqli, args.sqli_text_col, args.sqli_label_col, \"SQL Injection\"))\n    if args.xss:\n        parts.append(_load_binary(args.xss, args.xss_text_col, args.xss_label_col, \"XSS\"))\n    if args.normal:\n        n = read_csv_safely(args.normal, max_mb=200, max_rows=2_000_000)\n        require_columns(n, [args.normal_text_col])\n        parts.append(pd.DataFrame({config.TEXT_COL: n[args.normal_text_col].astype(str),\n                                   config.TEXT_LABEL_COL: \"Normal\"}))\n    if not parts:\n        raise SystemExit(\"Укажите хотя бы один из --sqli / --xss / --normal\")\n\n    df = pd.concat(parts, ignore_index=True).drop_duplicates()\n    conflicts = df.groupby(config.TEXT_COL)[config.TEXT_LABEL_COL].nunique()\n    df = df[~df[config.TEXT_COL].isin(conflicts[conflicts > 1].index)]\n\n    if args.max_per_class:\n        df = (df.groupby(config.TEXT_LABEL_COL, group_keys=False)\n                .apply(lambda g: g.sample(min(len(g), args.max_per_class), random_state=config.RANDOM_STATE)))\n    df = df.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)\n\n    if args.add_timestamps:\n        rng = np.random.default_rng(config.RANDOM_STATE)\n        ts = pd.Timestamp(\"2026-09-01 08:00\") + pd.to_timedelta(\n            np.sort(rng.uniform(0, 8 * 3600, len(df))), unit=\"s\")\n        df[config.TEXT_TIME_COL] = ts.strftime(\"%Y-%m-%d %H:%M:%S\")\n    return df\n\ndef main() -> None:\n    p = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter)\n    p.add_argument(\"--sqli\"); p.add_argument(\"--sqli-text-col\", default=\"Query\")\n    p.add_argument(\"--sqli-label-col\", default=\"Label\")\n    p.add_argument(\"--xss\"); p.add_argument(\"--xss-text-col\", default=\"Sentence\")\n    p.add_argument(\"--xss-label-col\", default=\"Label\")\n    p.add_argument(\"--normal\"); p.add_argument(\"--normal-text-col\", default=\"request\")\n    p.add_argument(\"--max-per-class\", type=int, default=0)\n    p.add_argument(\"--add-timestamps\", action=\"store_true\")\n    p.add_argument(\"--out\", default=str(config.TEXT_DATA_PATH))\n    args = p.parse_args()\n\n    df = build_dataset(args)\n    Path(args.out).parent.mkdir(parents=True, exist_ok=True)\n    df.to_csv(args.out, index=False)\n    print(f\"Записано {len(df)} строк в {args.out}\")\n    print(df[config.TEXT_LABEL_COL].value_counts().to_string())\n\nif __name__ == \"__main__\":\n    main()\n",
  "data/fp_benign_requests.csv": "request,label,note\n/search?q=o'reilly+python+book,Normal,апостроф в фамилии\n/search?q=select+a+size+guide,Normal,слово select\n/search?q=javascript+tutorial+for+beginners,Normal,слово javascript\n/search?q=drop+shipping+suppliers,Normal,слово drop\n/search?q=union+jack+flag,Normal,слово union\n/search?q=script+writing+course,Normal,слово script\n/search?q=rock+%26+roll+records,Normal,URL-кодированный амперсанд\n/search?q=ben+%26+jerry%27s,Normal,кодированный апостроф\n/search?q=how+to+delete+my+account,Normal,слово delete\n/search?q=insert+coin+arcade+machine,Normal,слово insert\n/search?q=update+drivers+windows+11,Normal,слово update\n/search?q=c%2B%2B+vs+java,Normal,кодированные плюсы\n/search?q=html+table+example,Normal,слово html\n/search?q=%3C3+valentine+card,Normal,символ <\n/search?q=price+%3C+100+and+rating+%3E+4,Normal,сравнения < >\n/profile?user=o'connor&tab=settings,Normal,апостроф в логине\ncomment=Great product -- would buy again!&rating=5,Normal,двойной дефис\ncomment=Size 10 = perfect fit; color as shown,Normal,знаки = и ;\n/api/v1/orders/1024?fields=id%2Cstatus,Normal,кодированная запятая\n/blog/how-to-choose-a-laptop?utm_source=newsletter,Normal,обычная ссылка\n\"/search?q=\"\"best laptop 2026\"\"\",Normal,кавычки\n/search?q=alert+bracelet+medical,Normal,слово alert\n/search?q=1%3D1+math+identity,Normal,1=1 в обычном тексте\n/search?q=where+to+buy+tables+and+chairs,Normal,слова where и tables\n",
  "tests/conftest.py": "import sys\nfrom pathlib import Path\n\nimport numpy as np\nimport pandas as pd\nimport pytest\n\nROOT = Path(__file__).resolve().parents[1]\nsys.path.insert(0, str(ROOT))\nsys.path.insert(0, str(ROOT / \"scripts\"))\n\n@pytest.fixture(scope=\"session\")\ndef network_df():\n    from generate_network_data import make_network\n    return make_network(3000, seed=3, start=\"2026-09-01 08:00\")\n\n@pytest.fixture(scope=\"session\")\ndef placeholder_text_df():\n    rng = np.random.default_rng(0)\n    words = [\"catalog\", \"item\", \"page\", \"search\", \"view\", \"order\", \"user\", \"blog\"]\n    markers = {\"Normal\": \"\", \"SQL Injection\": \"zzqmarker_s\", \"XSS\": \"zzqmarker_x\"}\n    rows = []\n    for label, marker in markers.items():\n        for i in range(120):\n            body = \" \".join(rng.choice(words, 4))\n            rows.append({\"request\": f\"/{body.replace(' ', '/')}?id={i} {marker}\".strip(),\n                         \"label\": label})\n    return pd.DataFrame(rows)\n",
  "tests/test_io_utils.py": "import pandas as pd\nimport pytest\n\nimport config\nfrom src.common.io_utils import (\n    DataValidationError, make_events_frame, read_csv_safely, require_columns,\n    sanitize_for_csv_export, validate_events,\n)\n\ndef test_rejects_non_csv_extension():\n    with pytest.raises(DataValidationError):\n        read_csv_safely(b\"a,b\\n1,2\\n\", filename=\"data.xlsx\")\n\ndef test_rejects_too_large_file():\n    big = b\"a\\n\" + b\"1\\n\" * 300_000\n    with pytest.raises(DataValidationError):\n        read_csv_safely(big, filename=\"x.csv\", max_mb=0.1)\n\ndef test_rejects_too_many_rows():\n    with pytest.raises(DataValidationError):\n        read_csv_safely(b\"a\\n\" + b\"1\\n\" * 50, filename=\"x.csv\", max_rows=10)\n\ndef test_strips_column_names():\n    df = read_csv_safely(b\" Flow Duration, Label\\n1,BENIGN\\n\", filename=\"x.csv\")\n    assert list(df.columns) == [\"Flow Duration\", \"Label\"]\n\ndef test_missing_columns_reported():\n    with pytest.raises(DataValidationError, match=\"request\"):\n        require_columns(pd.DataFrame({\"x\": [1]}), [\"request\"])\n\ndef test_csv_formula_injection_escaped():\n    df = pd.DataFrame({\"t\": [\"=1+1\", \"@cmd\", \"normal\", \"-x\"], \"n\": [-1, 2, 3, 4]})\n    out = sanitize_for_csv_export(df)\n    assert out[\"t\"].tolist() == [\"'=1+1\", \"'@cmd\", \"normal\", \"'-x\"]\n    assert out[\"n\"].tolist() == [-1, 2, 3, 4]\n\ndef _event(**over):\n    base = dict(event_id=[\"e1\"], timestamp=[\"\"], source_module=[\"http\"], model_name=[\"m\"],\n                prediction=[\"Normal\"], score=[0.9], is_alert=[False], needs_review=[False],\n                ground_truth=[\"\"], details=[\"d\"], status=[config.PREDICTION_STATUS])\n    base.update(over)\n    return base\n\ndef test_event_contract_complete():\n    df = make_events_frame(**_event())\n    assert list(df.columns) == config.EVENT_COLUMNS\n    validate_events(df)\n\ndef test_event_contract_missing_field():\n    data = _event(); data.pop(\"status\")\n    with pytest.raises(ValueError):\n        make_events_frame(**data)\n\ndef test_event_contract_bad_score():\n    with pytest.raises(DataValidationError):\n        validate_events(make_events_frame(**_event(score=[1.5])))\n",
  "tests/test_network.py": "import numpy as np\n\nimport config\nfrom src.common.io_utils import validate_events\nfrom src.network.models import analyze_network, network_events, train_network_models\nfrom src.network.preprocess import clean_network_data\n\ndef test_cleaning_removes_inf_and_duplicates(network_df):\n    clean, report = clean_network_data(network_df)\n    assert report[\"duplicates\"] > 0 and report[\"duplicates_removed\"] == report[\"duplicates\"]\n    assert not np.isinf(clean[config.NETWORK_FEATURES].to_numpy()).any()\n    assert not clean.duplicated().any()\n\ndef test_training_and_events(network_df):\n    clean, _ = clean_network_data(network_df)\n    models, metrics = train_network_models(clean)\n    rf = metrics[\"random_forest\"]\n\n    assert rf[\"f1_macro\"] > 0.7\n\n    iso = metrics[\"isolation_forest\"]\n    assert iso[\"false_positive_rate\"] < 0.3\n    assert iso[\"detection_rate_by_label\"][\"PortScan\"] > 0.8\n\n    analyzed = analyze_network(clean.head(200), models[\"isoforest\"], models[\"random_forest\"])\n    assert analyzed[\"if_score\"].between(0, 1).all()\n    ev = network_events(analyzed)\n    validate_events(ev)\n    assert len(ev) == 200 and set(ev[\"prediction\"]) <= {\"Anomaly\", \"Normal\"}\n    assert ev[\"timestamp\"].str.len().gt(0).all()\n\ndef test_works_without_label_and_timestamp(network_df):\n    clean, _ = clean_network_data(network_df)\n    models, _ = train_network_models(clean)\n    unlabeled = clean.drop(columns=[\"Label\", \"Timestamp\"]).head(50)\n    ev = network_events(analyze_network(unlabeled, models[\"isoforest\"]))\n    validate_events(ev)\n    assert (ev[\"ground_truth\"] == \"\").all()\n",
  "tests/test_nlp.py": "import numpy as np\nimport pandas as pd\n\nimport config\nfrom src.common.io_utils import validate_events\nfrom src.nlp.models import classify_requests, explain_request, http_events, train_nlp_models\nfrom src.nlp.preprocess import clean_text_data, normalize_label, normalize_request\n\ndef test_normalize_decodes_url_and_html():\n    assert normalize_request(\"Hello%20World\") == \"hello world\"\n    assert normalize_request(\"a%2520b\") == \"a b\"\n    assert normalize_request(\"&lt;b&gt;bold&lt;/b&gt;\") == \"<b>bold</b>\"\n    assert normalize_request(\"id=12345\") == \"id=0\"\n    assert normalize_request(None) == \"\"\n    assert len(normalize_request(\"x\" * 10_000)) == config.MAX_REQUEST_LENGTH\n\ndef test_special_characters_preserved():\n    s = normalize_request(\"q=it's;a<b>(c)--d/*e*/\")\n    for ch in \"';<>()-/*\":\n        assert ch in s\n\ndef test_label_aliases():\n    assert normalize_label(\"sqli\") == \"SQL Injection\"\n    assert normalize_label(\"benign\") == \"Normal\"\n    assert normalize_label(\" XSS \") == \"XSS\"\n\ndef test_dedup_after_normalization():\n    df = pd.DataFrame({\"request\": [\"/page?id=1\", \"/page?id=2\", \"/page%3Fid%3D3\", \"\"],\n                       \"label\": [\"Normal\"] * 4})\n    clean, rep = clean_text_data(df)\n    assert len(clean) == 1 and rep[\"duplicates_removed\"] == 2 and rep[\"empty_removed\"] == 1\n\ndef test_pipeline_mechanics(placeholder_text_df):\n    clean, _ = clean_text_data(placeholder_text_df)\n    models, metrics = train_nlp_models(clean)\n    for name in (\"logistic_regression\", \"naive_bayes\"):\n        m = metrics[name]\n        assert m[\"labels\"] == config.TEXT_CLASSES\n        assert np.array(m[\"confusion_matrix\"]).shape == (3, 3)\n        assert m[\"f1_macro\"] > 0.9\n\n    lr = models[\"logistic_regression\"]\n    res = classify_requests(lr, [\"/catalog/item?id=5\", \"/catalog zzqmarker_x\"])\n    prob_cols = [f\"prob_{c}\" for c in lr.classes_]\n    assert np.allclose(res[prob_cols].sum(axis=1), 1, atol=1e-3)\n    assert res.loc[1, \"prediction\"] == \"XSS\"\n    assert not explain_request(lr, \"/catalog zzqmarker_x\").empty\n\n    ev = http_events(pd.DataFrame({\"request\": res[\"request\"]}), res)\n    validate_events(ev)\n    assert ev[\"is_alert\"].tolist() == (res[\"prediction\"] != \"Normal\").tolist()\n\ndef test_review_flag_threshold(placeholder_text_df):\n    clean, _ = clean_text_data(placeholder_text_df)\n    models, _ = train_nlp_models(clean)\n    res = classify_requests(models[\"logistic_regression\"], [\"completely unrelated text ###\"])\n    assert res.loc[0, \"needs_review\"] == (res.loc[0, \"confidence\"] < config.NLP_REVIEW_THRESHOLD)\n",
  "tests/test_integration.py": "import pandas as pd\n\nimport config\nfrom src.common.io_utils import load_events, save_events, validate_events\nfrom src.network.models import analyze_network, network_events, train_network_models\nfrom src.network.preprocess import clean_network_data\nfrom src.nlp.models import classify_requests, http_events, train_nlp_models\nfrom src.nlp.preprocess import clean_text_data\n\ndef test_events_from_both_modules_merge(tmp_path, network_df, placeholder_text_df):\n    clean_net, _ = clean_network_data(network_df)\n    net_models, _ = train_network_models(clean_net)\n    net_ev = network_events(analyze_network(clean_net.head(100), net_models[\"isoforest\"],\n                                            net_models[\"random_forest\"]))\n\n    clean_txt, _ = clean_text_data(placeholder_text_df)\n    nlp_models, _ = train_nlp_models(clean_txt)\n    res = classify_requests(nlp_models[\"logistic_regression\"], clean_txt[\"request\"].head(30))\n    http_ev = http_events(clean_txt.head(30), res)\n\n    save_events(net_ev, tmp_path / \"n.csv\"); save_events(http_ev, tmp_path / \"h.csv\")\n    merged = pd.concat([load_events(tmp_path / \"n.csv\"), load_events(tmp_path / \"h.csv\")],\n                       ignore_index=True)\n    validate_events(merged)\n    assert list(merged.columns) == config.EVENT_COLUMNS\n    assert merged[\"event_id\"].is_unique\n    assert set(merged[\"source_module\"]) == {\"network\", \"http\"}\n    assert (merged[\"status\"] == config.PREDICTION_STATUS).all()\n",
  "tests/test_false_positives.py": "import joblib\nimport pandas as pd\nimport pytest\n\nimport config\nfrom src.nlp.models import classify_requests\n\nFP_FILE = config.DATA_DIR / \"fp_benign_requests.csv\"\n\n@pytest.mark.skipif(not config.NLP_MODEL_PATH.exists(), reason=\"NLP-модель ещё не обучена\")\ndef test_hard_benign_false_positive_rate():\n    model = joblib.load(config.NLP_MODEL_PATH)\n    df = pd.read_csv(FP_FILE)\n    res = classify_requests(model, df[\"request\"])\n    fpr = (res[\"prediction\"] != \"Normal\").mean()\n    print(\"\\nЛожные срабатывания:\\n\", res.loc[res[\"prediction\"] != \"Normal\",\n                                            [\"request\", \"prediction\", \"confidence\"]].to_string())\n    print(f\"FPR на трудных нормальных запросах: {fpr:.2%}\")\n    assert fpr <= 0.25\n",
  "scripts/generate_text_data.py": "from __future__ import annotations\n\nimport sys\nfrom pathlib import Path\nfrom urllib.parse import quote, quote_plus\n\nimport numpy as np\nimport pandas as pd\n\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\nimport config\n\ndef _build_normal_requests(rng: np.random.Generator, n: int) -> list[str]:\n    endpoints = [\n        \"/search\", \"/catalog/items\", \"/products/view\", \"/blog/post\",\n        \"/api/v1/articles\", \"/shop/filter\", \"/courses/list\", \"/help/faq\",\n        \"/profile/view\", \"/orders/history\", \"/forum/thread\", \"/docs/guide\",\n        \"/library/books\", \"/store/electronics\", \"/events/schedule\",\n    ]\n    adjectives = [\n        \"modern\", \"advanced\", \"beginner\", \"practical\", \"complete\", \"classic\",\n        \"wireless\", \"portable\", \"compact\", \"professional\", \"essential\", \"daily\",\n        \"secure\", \"fast\", \"clean\", \"custom\", \"official\", \"regional\", \"global\",\n    ]\n    nouns = [\n        \"python\", \"programming\", \"handbook\", \"laptop\", \"monitor\", \"keyboard\",\n        \"headphones\", \"camera\", \"backpack\", \"jacket\", \"shoes\", \"coffee\",\n        \"furniture\", \"guitar\", \"bicycle\", \"watch\", \"cookbook\", \"encyclopedia\",\n        \"analytics\", \"database\", \"network\", \"router\", \"workshop\", \"tutorial\",\n    ]\n    tricky_phrases = [\n        \"select best size for running shoes\",\n        \"how to select a monitor for home office\",\n        \"select delivery option and payment method\",\n        \"union station train schedule\",\n        \"european union travel guide\",\n        \"credit union mortgage calculator\",\n        \"union jack vintage poster\",\n        \"javascript course for beginners\",\n        \"learn javascript and css layout\",\n        \"html table styling tutorial\",\n        \"html form accessibility guide\",\n        \"script writing masterclass for cinema\",\n        \"calligraphy script font collection\",\n        \"theater script archive drama\",\n        \"drop shipping business guide\",\n        \"eye drop bottle travel size\",\n        \"rain drop photography tips\",\n        \"how to delete old cache files\",\n        \"delete duplicate photos on phone\",\n        \"insert coin retro arcade cabinet\",\n        \"shoe insert orthopedic arch support\",\n        \"update graphics drivers on windows\",\n        \"software update release notes\",\n        \"weather alert notification settings\",\n        \"medical alert wristband silver\",\n        \"where to buy dining tables and chairs\",\n        \"wooden tables for kitchen\",\n        \"o'reilly learning library subscription\",\n        \"o'connor family bakery dublin\",\n        \"o'neill wetsuit surfing gear\",\n        \"mcdonald's breakfast menu nutrition\",\n        \"ben & jerry's ice cream flavors\",\n        \"rock & roll hall of fame tickets\",\n        \"c++ vs rust performance benchmark\",\n        \"math identity a=a and 1=1 explanation\",\n        \"filter price < 200 and rating > 4\",\n        \"temperature < 0 winter coat\",\n    ]\n    comments = [\n        \"Great quality -- arrived on time and works as expected!\",\n        \"Fast shipping -- would definitely recommend to friends\",\n        \"Size L = great fit; material is soft and durable\",\n        \"Color = navy blue; packaging was neat\",\n        \"Battery life > 10 hours -- very happy with this purchase\",\n        \"Rating 5/5 -- best value under $100\",\n    ]\n    surnames = [\n        \"o'reilly\", \"o'connor\", \"o'neill\", \"o'brien\", \"d'arcy\",\n        \"smith\", \"johnson\", \"williams\", \"brown\", \"miller\", \"wilson\",\n    ]\n    tabs = [\"settings\", \"orders\", \"wishlist\", \"notifications\", \"security\", \"billing\"]\n\n    requests: list[str] = []\n    while len(requests) < n:\n        mode = int(rng.integers(0, 6))\n        ep = rng.choice(endpoints)\n        adj = rng.choice(adjectives)\n        noun = rng.choice(nouns)\n        noun2 = rng.choice(nouns)\n        if mode == 0:\n            q = f\"{adj} {noun} for {noun2}\"\n            if rng.random() < 0.4:\n                q = quote_plus(q)\n            requests.append(f\"{ep}?q={q}&sort={rng.choice(['relevance', 'price_asc', 'rating', 'newest'])}\")\n        elif mode == 1:\n            phrase = rng.choice(tricky_phrases)\n            suffix = f\" {adj} {noun}\" if rng.random() < 0.75 else f\" {noun2}\"\n            q = phrase + suffix\n            if rng.random() < 0.5:\n                q = quote_plus(q)\n            requests.append(f\"{ep}?q={q}\")\n        elif mode == 2:\n            c = rng.choice(comments)\n            requests.append(f\"comment={c} ({adj} {noun} {noun2})&rating=5\")\n        elif mode == 3:\n            user = rng.choice(surnames)\n            tab = rng.choice(tabs)\n            requests.append(f\"/profile?user={user}_{adj}&tab={tab}&view={noun}\")\n        elif mode == 4:\n            fields = rng.choice([\"id,status,total\", \"title,author,category\", \"name,price,stock\"])\n            if rng.random() < 0.5:\n                fields = quote(fields)\n            requests.append(f\"/api/v1/{noun}/{adj}/{noun2}?fields={fields}&format=json\")\n        else:\n            requests.append(f\"/blog/{adj}-{noun}-review-and-{noun2}-guide?utm_source={rng.choice(['newsletter', 'organic', 'rss', 'partner'])}\")\n    return requests[:n]\n\ndef _build_sqli_requests(rng: np.random.Generator, n: int) -> list[str]:\n    endpoints = [\n        \"/login\", \"/auth/signin\", \"/products\", \"/catalog/search\",\n        \"/users/profile\", \"/api/v1/items\", \"/orders/lookup\", \"/admin/audit\",\n    ]\n    params = [\"id\", \"user\", \"username\", \"query\", \"cat\", \"item_id\", \"order_ref\", \"filter\"]\n    tables = [\n        \"users\", \"accounts\", \"customers\", \"admin_users\", \"credentials\",\n        \"orders\", \"payments\", \"sessions\", \"members\", \"staff_records\",\n    ]\n    cols = [\n        \"username,password\", \"email,passwd_hash\", \"user_id,token\",\n        \"login,secret_key\", \"card_number,cvv\", \"role,api_key\",\n    ]\n    payload_templates = [\n        \"' OR '1'='1' -- {tag}\",\n        \"' OR 1=1 -- {tag}\",\n        \"admin' -- {tag}\",\n        \"' OR 'a'='a' /* {tag} */\",\n        \"' UNION SELECT {cols} FROM {table} -- {tag}\",\n        \"' UNION ALL SELECT NULL,{cols} FROM {table}# {tag}\",\n        \"1'; DROP TABLE {table}; -- {tag}\",\n        \"1'; UPDATE {table} SET role='admin' WHERE username='{tag}'; --\",\n        \"1' AND (SELECT COUNT(*) FROM {table}) > 0 -- {tag}\",\n        \"' AND SLEEP(5) AND '{tag}'='{tag}\",\n        \"'; WAITFOR DELAY '0:0:5' -- {tag}\",\n        \"' OR EXISTS(SELECT 1 FROM {table} WHERE {cols_first} LIKE '%{tag}%') --\",\n        \"1' ORDER BY 1,2,3 -- {tag}\",\n        \"' UNION SELECT table_name,column_name FROM information_schema.columns WHERE '{tag}'='{tag}\",\n    ]\n    tags = [\n        \"alpha\", \"beta\", \"gamma\", \"delta\", \"omega\", \"sigma\", \"theta\", \"kappa\",\n        \"zeta\", \"lambda\", \"auth\", \"check\", \"probe\", \"test\", \"admin\", \"root\",\n        \"dump\", \"extract\", \"audit\", \"verify\", \"login\", \"bypass\", \"access\",\n    ]\n    requests: list[str] = []\n    while len(requests) < n:\n        ep = rng.choice(endpoints)\n        p = rng.choice(params)\n        tbl = rng.choice(tables)\n        col_pair = rng.choice(cols)\n        tag = f\"{rng.choice(tags)}_{rng.choice(tags)}\"\n        tpl = rng.choice(payload_templates)\n        payload = tpl.format(\n            table=tbl,\n            cols=col_pair,\n            cols_first=col_pair.split(\",\")[0],\n            tag=tag,\n        )\n        if rng.random() < 0.25:\n            payload = payload.upper()\n        if rng.random() < 0.30:\n            payload = quote_plus(payload)\n        elif rng.random() < 0.10:\n            payload = quote_plus(quote_plus(payload))\n        requests.append(f\"{ep}?{p}={payload}\")\n    return requests[:n]\n\ndef _build_xss_requests(rng: np.random.Generator, n: int) -> list[str]:\n    endpoints = [\n        \"/search\", \"/comments/add\", \"/feedback/submit\", \"/profile/bio\",\n        \"/forum/reply\", \"/guestbook/sign\", \"/support/ticket\", \"/blog/comment\",\n    ]\n    params = [\"q\", \"comment\", \"message\", \"text\", \"name\", \"title\", \"bio\", \"return_url\"]\n    js_targets = [\n        \"document.cookie\", \"document.domain\", \"window.origin\",\n        \"localStorage.getItem('token')\", \"'xss_test'\", \"window.location\",\n    ]\n    tags_pool = [\n        \"header\", \"footer\", \"sidebar\", \"modal\", \"banner\", \"avatar\", \"widget\",\n        \"preview\", \"card\", \"tooltip\", \"input\", \"form\", \"navbar\", \"content\",\n    ]\n    payload_templates = [\n        \"<script>alert({target})</script><!-- {tag} -->\",\n        \"<ScRiPt>confirm({target});</sCrIpT>/{tag}\",\n        '\"><script src=\"https://evil.example/{tag}.js\"></script>',\n        \"<img src=x onerror=alert({target}) alt={tag}>\",\n        \"<svg/onload=alert({target}) id={tag}>\",\n        \"<body onload=prompt({target}) data-ref={tag}>\",\n        '<iframe src=\"javascript:alert({target})\" title=\"{tag}\"></iframe>',\n        \"<input autofocus onfocus=alert({target}) name={tag}>\",\n        \"<details open ontoggle=confirm({target})>{tag}</details>\",\n        '<a href=\"javascript:eval(alert({target}))\">{tag}</a>',\n        '\\'><div onmouseover=\"alert({target})\">{tag}</div>',\n    ]\n    requests: list[str] = []\n    while len(requests) < n:\n        ep = rng.choice(endpoints)\n        p = rng.choice(params)\n        target = rng.choice(js_targets)\n        tag = f\"{rng.choice(tags_pool)}_{rng.choice(tags_pool)}\"\n        tpl = rng.choice(payload_templates)\n        payload = tpl.format(target=target, tag=tag)\n        r = rng.random()\n        if r < 0.20:\n            payload = payload.replace(\"<\", \"&lt;\").replace(\">\", \"&gt;\")\n        elif r < 0.45:\n            payload = quote_plus(payload)\n        elif r < 0.55:\n            payload = quote_plus(quote_plus(payload))\n        requests.append(f\"{ep}?{p}={payload}\")\n    return requests[:n]\n\ndef make_text_dataset(n_normal: int = 2600, n_sqli: int = 1800, n_xss: int = 1800, seed: int = 42) -> pd.DataFrame:\n    rng = np.random.default_rng(seed)\n    rows = (\n        [(r, \"Normal\") for r in _build_normal_requests(rng, n_normal)]\n        + [(r, \"SQL Injection\") for r in _build_sqli_requests(rng, n_sqli)]\n        + [(r, \"XSS\") for r in _build_xss_requests(rng, n_xss)]\n    )\n    df = pd.DataFrame(rows, columns=[config.TEXT_COL, config.TEXT_LABEL_COL])\n    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)\n    ts = pd.Timestamp(\"2026-09-01 08:00\") + pd.to_timedelta(\n        np.sort(rng.uniform(0, 8 * 3600, len(df))), unit=\"s\"\n    )\n    df[config.TEXT_TIME_COL] = ts.strftime(\"%Y-%m-%d %H:%M:%S\")\n    return df\n\ndef main() -> None:\n    config.DATA_DIR.mkdir(parents=True, exist_ok=True)\n    df = make_text_dataset()\n    df.to_csv(config.TEXT_DATA_PATH, index=False)\n    df_new = make_text_dataset(n_normal=200, n_sqli=100, n_xss=100, seed=99)\n    df_new.to_csv(config.DATA_DIR / \"http_requests_new.csv\", index=False)\n    print(f\"Текстовые данные записаны в {config.TEXT_DATA_PATH} ({len(df)} строк)\")\n    print(df[config.TEXT_LABEL_COL].value_counts().to_string())\n\nif __name__ == \"__main__\":\n    main()\n"
}

for rel_path, file_text in EMBEDDED_PROJECT_FILES.items():
    dest = PROJECT_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(file_text, encoding="utf-8")

os.chdir(PROJECT_ROOT)
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import pytest

try:
    from IPython.display import Image as IPyImage, display
except ImportError:
    IPyImage = None
    display = print

def show_fig(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if IPyImage is not None:
        display(IPyImage(data=buf.getvalue()))

import config
import train_all
from generate_network_data import main as generate_network_main
from generate_text_data import main as generate_text_main
from src.network.preprocess import load_network_csv, clean_network_data
from src.network.models import train_network_models, analyze_network
from src.nlp.preprocess import normalize_request, load_text_csv, clean_text_data
from src.nlp.models import train_nlp_models, classify_requests, explain_request
from src.common.io_utils import load_events, validate_events

sns.set_theme(style="whitegrid")

generate_network_main()
generate_text_main()

raw_net_df = load_network_csv(config.NETWORK_DATA_PATH, require_label=True)
net_df, net_quality = clean_network_data(raw_net_df)
display(pd.DataFrame([net_quality]))
display(net_df.head())
display(net_df.groupby(config.NETWORK_LABEL_COL)[config.NETWORK_FEATURES].median().T.round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
vc = net_df[config.NETWORK_LABEL_COL].value_counts()
sns.barplot(x=vc.values, y=vc.index, ax=axes[0], hue=vc.index, palette="viridis", legend=False)
axes[0].set_title("Распределение классов трафика (N = 12 000)", fontweight="bold")
axes[0].set_xlabel("Количество потоков")
for i, v in enumerate(vc.values):
    axes[0].text(v + 120, i, f"{v} ({v/len(net_df):.1%})", va="center", fontsize=9, fontweight="bold")
axes[0].set_xlim(0, max(vc.values) * 1.22)

sns.boxplot(data=net_df, x=config.NETWORK_LABEL_COL, y="Total Fwd Packets", ax=axes[1], hue=config.NETWORK_LABEL_COL, palette="Set2", legend=False)
axes[1].set_yscale("log")
axes[1].set_title("Total Fwd Packets (log-шкала)", fontweight="bold")
axes[1].tick_params(axis="x", rotation=25)

sns.boxplot(data=net_df, x=config.NETWORK_LABEL_COL, y="Flow Packets/s", ax=axes[2], hue=config.NETWORK_LABEL_COL, palette="Set2", legend=False)
axes[2].set_yscale("symlog")
axes[2].set_title("Flow Packets/s (symlog-шкала)", fontweight="bold")
axes[2].tick_params(axis="x", rotation=25)
fig.tight_layout()
show_fig(fig)

fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(net_df[config.NETWORK_FEATURES].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("Матрица корреляций 14 числовых признаков сетевого потока (CIC-IDS2017)", fontsize=13, fontweight="bold")
fig.tight_layout()
show_fig(fig)

net_models, net_metrics = train_network_models(net_df)
iso_m = net_metrics["isolation_forest"]
rf_m = net_metrics["random_forest"]

summary_net = pd.DataFrame([
    {
        "Модель": f"Isolation Forest (contamination={iso_m['contamination']})",
        "Accuracy": iso_m["accuracy"],
        "Precision (macro)": iso_m["precision_macro"],
        "Recall (macro)": iso_m["recall_macro"],
        "F1 (macro)": iso_m["f1_macro"],
        "FPR": iso_m["false_positive_rate"],
    },
    {
        "Модель": f"Random Forest {rf_m['best_params']}",
        "Accuracy": rf_m["accuracy"],
        "Precision (macro)": rf_m["precision_macro"],
        "Recall (macro)": rf_m["recall_macro"],
        "F1 (macro)": rf_m["f1_macro"],
        "FPR": rf_m["false_positive_rate"],
    },
])
display(summary_net)

analyzed_net = analyze_network(net_df, net_models["isoforest"], net_models["random_forest"])

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
det_rates = pd.Series(iso_m["detection_rate_by_label"]).sort_values(ascending=True)
det_rates.plot.barh(ax=axes[0, 0], color=["#C44E52" if idx != "BENIGN" else "#4C72B0" for idx in det_rates.index])
axes[0, 0].set_title("Isolation Forest: доля помеченных записей по фактической метке", fontweight="bold")
for i, v in enumerate(det_rates.values):
    axes[0, 0].text(v + 0.015, i, f"{v:.1%}", va="center", fontweight="bold")
axes[0, 0].set_xlim(0, 1.15)

fi = pd.Series(rf_m["feature_importance"]).sort_values(ascending=True)
fi.plot.barh(ax=axes[0, 1], color="#55A868")
axes[0, 1].set_title("Random Forest: важность признаков (Feature Importance)", fontweight="bold")

sns.histplot(data=analyzed_net, x="if_score", hue="if_anomaly", bins=40, ax=axes[1, 0], palette={False: "#4C72B0", True: "#C44E52"})
axes[1, 0].set_title("Распределение оценки аномальности (if_score)", fontweight="bold")

sample_net = analyzed_net.sample(min(3000, len(analyzed_net)), random_state=42)
sns.scatterplot(data=sample_net, x="Flow Duration", y="Flow Packets/s", hue="if_anomaly", s=16, ax=axes[1, 1], palette={False: "#4C72B0", True: "#C44E52"})
axes[1, 1].set_xscale("symlog")
axes[1, 1].set_yscale("symlog")
axes[1, 1].set_title("Потоки: длительность vs пакетов/с (красные — аномалии IF)", fontweight="bold")
fig.tight_layout()
show_fig(fig)

unusual_benign = analyzed_net[
    (analyzed_net["if_anomaly"] == True)
    & (analyzed_net["rf_prediction"] == "BENIGN")
    & (analyzed_net["Label"] == "BENIGN")
]
display(unusual_benign[["Timestamp", "Destination Port", "Flow Duration", "Total Fwd Packets", "if_score", "rf_prediction", "rf_confidence", "Label"]].head(5))

examples_raw = [
    "/login?user=admin%27%20OR%201%3D1%20--",
    "/search?q=%253Cscript%253Ealert(123)%253C%252Fscript%253E",
    "/comment?msg=&lt;img src=x onerror=alert(&#39;xss&#39;)&gt;",
    "/search?q=o'reilly+python+book&page=25",
]
display(pd.DataFrame({
    "Исходный запрос": examples_raw,
    "После normalize_request()": [normalize_request(x) for x in examples_raw],
}))

raw_txt_df = load_text_csv(config.TEXT_DATA_PATH, require_label=True)
txt_df, txt_quality = clean_text_data(raw_txt_df)
display(pd.DataFrame([txt_quality]))

nlp_models, nlp_metrics = train_nlp_models(txt_df)
lr_m = nlp_metrics["logistic_regression"]
nb_m = nlp_metrics["naive_bayes"]

summary_nlp = pd.DataFrame([
    {
        "Модель": f"TF-IDF + Logistic Regression {lr_m['best_params']}",
        "CV F1 (macro)": lr_m["cv_f1_macro"],
        "Test Accuracy": lr_m["accuracy"],
        "Precision (macro)": lr_m["precision_macro"],
        "Recall (macro)": lr_m["recall_macro"],
        "Test F1 (macro)": lr_m["f1_macro"],
        "FPR": lr_m["false_positive_rate"],
    },
    {
        "Модель": f"TF-IDF + Multinomial Naive Bayes {nb_m['best_params']}",
        "CV F1 (macro)": nb_m["cv_f1_macro"],
        "Test Accuracy": nb_m["accuracy"],
        "Precision (macro)": nb_m["precision_macro"],
        "Recall (macro)": nb_m["recall_macro"],
        "Test F1 (macro)": nb_m["f1_macro"],
        "FPR": nb_m["false_positive_rate"],
    },
])
display(summary_nlp)

demo_queries = [
    "/search?q=o'reilly+python+book",
    "/products?id=105%27%20UNION%20SELECT%20username,password%20FROM%20users--",
    "/feedback?message=%3Csvg%2Fonload%3Dalert(document.cookie)%3E",
    "completely unrelated ambiguous text ###",
]
lr_pipe = nlp_models["logistic_regression"]
display(classify_requests(lr_pipe, demo_queries))

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
for ax, q, title_lbl, color in [
    (axes[0], demo_queries[1], "Топ-8 символьных n-грамм: SQL Injection", "#1E3A8A"),
    (axes[1], demo_queries[2], "Топ-8 символьных n-грамм: XSS", "#0D9488"),
]:
    exp_df = explain_request(lr_pipe, q, top_k=8).iloc[::-1]
    ax.barh(exp_df["ngram"], exp_df["contribution"], color=color)
    ax.set_title(title_lbl, fontweight="bold")
    ax.set_xlabel("Вклад признака (TF-IDF × вес модели)")
for q in demo_queries[1:3]:
    display(explain_request(lr_pipe, q, top_k=8))
fig.tight_layout()
show_fig(fig)

sys.argv = ["train_all.py"]
train_all.main()

cm_files = [
    ("cm_network_isolation_forest.png", "Сеть: Isolation Forest"),
    ("cm_network_random_forest.png", "Сеть: Random Forest"),
    ("cm_nlp_logistic_regression.png", "NLP: Logistic Regression"),
    ("cm_nlp_naive_bayes.png", "NLP: Multinomial Naive Bayes"),
]
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (fname, title) in zip(axes.flat, cm_files):
    img_path = config.REPORTS_DIR / fname
    if img_path.exists():
        ax.imshow(mpimg.imread(img_path))
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
fig.tight_layout()
show_fig(fig)

fp_df = pd.read_csv(config.DATA_DIR / "fp_benign_requests.csv")
trained_lr = joblib.load(config.NLP_MODEL_PATH)
fp_results = classify_requests(trained_lr, fp_df["request"])
fp_full = pd.concat([fp_df[["note"]], fp_results], axis=1)
fpr_hard = (fp_full["prediction"] != "Normal").mean()
print(f"FPR (fp_benign_requests.csv): {fpr_hard:.2%}")
display(fp_full)

net_ev = load_events(config.NETWORK_EVENTS_PATH)
http_ev = load_events(config.HTTP_EVENTS_PATH)
all_events = pd.concat([net_ev, http_ev], ignore_index=True)
validate_events(all_events)

fig = plt.figure(figsize=(16, 8.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.2])

ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.axis("off")
kpis = [
    ("Сетевых записей", f"{len(net_ev):,}".replace(",", " "), "#1E3A8A"),
    ("Аномалий (IF)", str(int(net_ev["is_alert"].sum())), "#C44E52"),
    ("Запросов Normal", str(int((http_ev["prediction"] == "Normal").sum())), "#2E7D32"),
    ("SQL Injection", str(int((http_ev["prediction"] == "SQL Injection").sum())), "#D97706"),
    ("XSS", str(int((http_ev["prediction"] == "XSS").sum())), "#7C3AED"),
]
for idx, (label, val, col) in enumerate(kpis):
    x0 = idx * 0.20 + 0.01
    rect = plt.Rectangle((x0, 0.12), 0.18, 0.78, facecolor="#F8FAFC", edgecolor=col, linewidth=2, transform=ax_kpi.transAxes)
    ax_kpi.add_patch(rect)
    ax_kpi.text(x0 + 0.09, 0.62, val, ha="center", va="center", fontsize=18, fontweight="bold", color=col, transform=ax_kpi.transAxes)
    ax_kpi.text(x0 + 0.09, 0.28, label, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#334155", transform=ax_kpi.transAxes)
ax_kpi.set_title("AI Security Monitor — Сводная инфографика панели мониторинга (5 KPI)", fontsize=14, fontweight="bold", pad=8)

ax_g1 = fig.add_subplot(gs[1, 0])
net_counts = net_ev["prediction"].value_counts()
sns.barplot(x=net_counts.values, y=net_counts.index, ax=ax_g1, hue=net_counts.index, palette=["#4C72B0", "#C44E52"], legend=False)
ax_g1.set_title("Сетевой модуль: события", fontweight="bold")
ax_g1.set_xlabel("Количество")

ax_g2 = fig.add_subplot(gs[1, 1])
http_counts = http_ev["prediction"].value_counts()
sns.barplot(x=http_counts.values, y=http_counts.index, ax=ax_g2, hue=http_counts.index, palette="viridis", legend=False)
ax_g2.set_title("HTTP-модуль: прогнозы", fontweight="bold")
ax_g2.set_xlabel("Количество")

ax_g3 = fig.add_subplot(gs[1, 2])
models_comp = pd.DataFrame([
    {"Модель": "IsoForest", "F1-macro": iso_m["f1_macro"], "1 - FPR": 1 - iso_m["false_positive_rate"]},
    {"Модель": "RandomForest", "F1-macro": rf_m["f1_macro"], "1 - FPR": 1 - rf_m["false_positive_rate"]},
    {"Модель": "LogReg (NLP)", "F1-macro": lr_m["f1_macro"], "1 - FPR": 1 - lr_m["false_positive_rate"]},
    {"Модель": "NaiveBayes", "F1-macro": nb_m["f1_macro"], "1 - FPR": 1 - nb_m["false_positive_rate"]},
]).set_index("Модель")
models_comp.plot.bar(ax=ax_g3, color=["#1E3A8A", "#0D9488"], rot=15)
ax_g3.set_ylim(0, 1.12)
ax_g3.set_title("Сравнение F1-macro и (1 - FPR)", fontweight="bold")
fig.tight_layout()
show_fig(fig)

display(pd.DataFrame([{
    "1. Сетевых записей": len(net_ev),
    "2. Аномалий (IF)": int(net_ev["is_alert"].sum()),
    "3. Запросов Normal": int((http_ev["prediction"] == "Normal").sum()),
    "4. SQL Injection": int((http_ev["prediction"] == "SQL Injection").sum()),
    "5. XSS": int((http_ev["prediction"] == "XSS").sum()),
    "Всего событий": len(all_events),
    "Требуют проверки (needs_review)": int(all_events["needs_review"].sum()),
}]))

display(all_events.groupby("source_module").head(4).reset_index(drop=True))

pytest.main(["-v"])

archive_file = shutil.make_archive(
    str(WORK_ROOT / "ai_security_monitor_trained_bundle"),
    "zip",
    root_dir=PROJECT_ROOT.parent,
    base_dir=PROJECT_ROOT.name,
)
print("Архив проекта сохранён:", archive_file)

import re
import socket
import time
import urllib.request
import webbrowser

try:
    from IPython.display import HTML, Javascript
except ImportError:
    HTML = None
    Javascript = None

def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def _start_cloudflared_tunnel(port: int) -> str | None:

    cf_bin = Path("/tmp/cloudflared")
    try:
        if not cf_bin.exists():
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                str(cf_bin),
            )
            cf_bin.chmod(0o755)
        log_path = Path("/tmp/cloudflared.log")
        log_file = open(log_path, "w+", encoding="utf-8")
        subprocess.Popen(
            [str(cf_bin), "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        for _ in range(30):
            time.sleep(0.5)
            txt = log_path.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", txt)
            if m:
                return m.group(0)
    except Exception:
        return None
    return None

def launch_streamlit_site(port: int = 8501):

    subprocess.run(["pkill", "-f", "streamlit run"], check=False)
    subprocess.run(["pkill", "-f", "cloudflared tunnel"], check=False)
    time.sleep(0.5)

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py"),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--server.enableWebsocketCompression", "false",
        "--browser.gatherUsageStats", "false",
    ]
    log_path = PROJECT_ROOT / "streamlit_server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=log_file, stderr=subprocess.STDOUT)

    for _ in range(40):
        if _is_port_open(port):
            break
        time.sleep(0.5)

    in_colab = "google.colab" in sys.modules
    local_url = f"http://localhost:{port}"

    if in_colab:
        from google.colab import output
        colab_proxy_url = ""
        try:
            colab_proxy_url = str(output.eval_js(f"google.colab.kernel.proxyPort({port})")).strip()
        except Exception:
            pass

        public_url = _start_cloudflared_tunnel(port)
        primary_url = public_url or colab_proxy_url or local_url

        print("\n" + "=" * 78)
        print("🌐 ОТДЕЛЬНЫЙ САЙТ AI SECURITY MONITOR ЗАПУЩЕН!")
        if public_url:
            print(f"🔗 Внешний сайт (Cloudflare): {public_url}")
        if colab_proxy_url:
            print(f"🔗 Прокси-сайт Google Colab:  {colab_proxy_url}")
        print("=" * 78)

        try:
            output.serve_kernel_port_as_window(
                port,
                anchor_text="🚀 Нажмите здесь, чтобы открыть сайт AI Security Monitor в отдельном окне (Colab Proxy)",
            )
        except Exception:
            pass

        if HTML is not None:
            cf_btn = (
                f'<a href="{public_url}" target="_blank" rel="noopener" '
                f'style="display:inline-block;margin-right:12px;margin-top:6px;padding:11px 22px;'
                f'background:#0D9488;color:#ffffff;text-decoration:none;border-radius:8px;'
                f'font-weight:700;font-size:14px;">'
                f'🌐 Открыть внешний сайт ({public_url})</a>'
                if public_url else ""
            )
            proxy_btn = (
                f'<a href="{colab_proxy_url}" target="_blank" rel="noopener" '
                f'style="display:inline-block;margin-top:6px;padding:11px 22px;'
                f'background:#1E3A8A;color:#ffffff;text-decoration:none;border-radius:8px;'
                f'font-weight:700;font-size:14px;">'
                f'🛡️ Открыть через Google Colab Proxy</a>'
                if colab_proxy_url else ""
            )
            display(HTML(f"""
            <div style="margin:14px 0;padding:18px 22px;border:2px solid #0D9488;border-radius:12px;
                        background:#F0FDFA;font-family:sans-serif;">
                <div style="font-size:16px;font-weight:700;color:#0F172A;margin-bottom:6px;">
                    🛡️ Веб-приложение AI Security Monitor запущено на порту {port}
                </div>
                <div style="font-size:13.5px;color:#334155;margin-bottom:10px;">
                    Если браузер заблокировал автоматическое всплывающее окно, нажмите на любую из кнопок ниже —
                    откроется полноценный интерактивный сайт в новой вкладке:
                </div>
                {cf_btn}{proxy_btn}
            </div>
            """))
        if Javascript is not None and primary_url:
            display(Javascript(f'window.open("{primary_url}", "_blank");'))
        return proc

    print("\n" + "=" * 78)
    print(f"🌐 Сайт AI Security Monitor запущен и открывается в браузере: {local_url}")
    print("=" * 78)
    try:
        webbrowser.open(local_url, new=2)
    except Exception:
        pass

    if HTML is not None and IPyImage is not None:
        display(HTML(f"""
        <div style="margin:12px 0;padding:16px 20px;border:2px solid #1E3A8A;border-radius:10px;
                    background:#EFF6FF;font-family:sans-serif;">
            <b>🛡️ AI Security Monitor запущен:</b>
            <a href="{local_url}" target="_blank" rel="noopener"
               style="margin-left:12px;padding:9px 18px;background:#1E3A8A;color:#fff;
                      text-decoration:none;border-radius:6px;font-weight:700;">
                🚀 Открыть {local_url} в новой вкладке
            </a>
        </div>
        """))

    if IPyImage is None and sys.stdin is not None and sys.stdin.isatty() and os.environ.get("AI_SECURITY_NO_WAIT") != "1":
        print("Сервер работает. Нажмите Ctrl+C в терминале, чтобы остановить сайт...")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
    return proc

streamlit_proc = launch_streamlit_site(port=8501)
