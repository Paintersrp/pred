import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import { AgGridReact } from "ag-grid-react";
import { Link } from "react-router-dom";

import "ag-grid-community/styles/ag-grid.css";
import "./History.css";
import "ag-grid-community/styles/ag-theme-alpine.css";

const gridOptions = {
  onGridReady: (event) => event.api.sizeColumnsToFit(),
};

const Table = () => {
  const gridRef = useRef();
  const [rowData, setRowData] = useState();
  const [scoreData, setScoreData] = useState();

  const [columnDefs, setColumnDefs] = useState([
    { field: "Date", filter: true, minWidth: 125, flex: 1 },
    { field: "A_Team", minWidth: 200, flex: 1 },
    { field: "A_Odds", minWidth: 125, flex: 1 },
    { field: "H_Team", minWidth: 200, flex: 1 },
    { field: "H_Odds", minWidth: 125, flex: 1 },
    { field: "MOV", minWidth: 125, flex: 1 },
    { field: "Outcome", minWidth: 125, flex: 1 },
    { field: "Pred", minWidth: 125, flex: 1 },
  ]);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    resizable: true,
  }));

  const cellClickedListener = useCallback((event) => {
    console.log("cellClicked", event);
  }, []);

  useEffect(() => {
    fetch("./data/pred_history.json")
      .then((result) => result.json())
      .then((rowData) => setRowData(rowData));
  }, []);

  useEffect(() => {
    async function awaitFetch() {
      const response = fetch("./data/scoring.json").then((result) =>
        result.json()
      );
      const data = await response;
      setScoreData(data);
    }

    if (!scoreData) {
      awaitFetch();
    }
  }, []);

  return (
    <div className="layout-table text-white">
      {scoreData
        ? scoreData.map((obj) => (
            <div
              className="dark-table ag-theme-alpine bg-dark w-[1210px]"
              style={{ height: 900, fontSize: 12 }}
            >
              <div className="scoring flex flex-col">
                <h2 className="set-width align-center text-xl p-2 border set-bold">
                  Prediction History
                </h2>
                <div className="scoring mb-2 set-width flex-1 justify-center p-4">
                  <div className="flex flex-row align-center">
                    <label className="flex-1 flex-row align-center text-[13px] mr-4 mb-1 set-bold">
                      Correct
                    </label>
                    <label className="flex-1 flex-row align-center mr-4 text-[13px] set-bold">
                      Incorrect
                    </label>
                    <label className="flex-1 flex-row align-center text-[13px] highlight set-bold">
                      Ratio
                    </label>
                  </div>
                  <div className="flex flex-row align-center">
                    <label className="flex-1 flex-row align-center mr-4 border text-[13px] set-bold">
                      {obj.correct}
                    </label>
                    <label className="flex-1 flex-row align-center mr-4 border text-[13px] set-bold">
                      {obj.incorrect}
                    </label>
                    <label className="flex-1 flex-row align-center border text-[13px] highlight set-bold">
                      {obj.ratio}
                    </label>
                  </div>
                </div>
              </div>
              <AgGridReact
                ref={gridRef}
                rowData={rowData}
                columnDefs={columnDefs}
                defaultColDef={defaultColDef}
                rowHeight={30}
                animateRows={false}
                rowSelection="multiple"
                pagination={true}
                paginationPageSize={25}
                onCellClicked={cellClickedListener}
              />
            </div>
          ))
        : null}
    </div>
  );
};

export default Table;
