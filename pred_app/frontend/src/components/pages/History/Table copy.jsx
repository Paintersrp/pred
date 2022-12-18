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
import "./Table.css";
import "ag-grid-community/styles/ag-theme-alpine.css";

const gridOptions = {
  onGridReady: (event) => event.api.sizeColumnsToFit(),
};

const Table = () => {
  const gridRef = useRef();
  const [rowData, setRowData] = useState();
  const [typeState, setTypeState] = useState(1);

  const [columnDefs, setColumnDefs] = useState([
    { field: "Date", filter: true, minWidth: 125, flex: 1 },
    { field: "A_Team", minWidth: 200, flex: 1 },
    { field: "A_Odds", minWidth: 125, flex: 1 },
    { field: "H_Team", minWidth: 200, flex: 1 },
    { field: "H_Odds", minWidth: 125, flex: 1 },
    { field: "Actual", minWidth: 125, flex: 1 },
    { field: "Outcome", minWidth: 125, flex: 1 },
  ]);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    resizable: true,
  }));

  const cellClickedListener = useCallback((event) => {
    console.log("cellClicked", event);
  }, []);

  const handleClickPerGame = (e) => {
    setTypeState(2);
    fetch("./data/pred_history.json")
      .then((result) => result.json())
      .then((rowData) => setRowData(rowData));
  };

  const handleClickPer100 = (e) => {
    setTypeState(1);
    fetch("./data/pred_history.json")
      .then((result) => result.json())
      .then((rowData) => setRowData(rowData));
  };

  useEffect(() => {
    fetch("./data/pred_history.json")
      .then((result) => result.json())
      .then((rowData) => setRowData(rowData));
  }, []);

  return (
    <div className="layout-table text-white">
      <div className="stats-select">
        <div className="stats-select">
          <Link to="/stats" className="selected btn-table">
            <button className="text-[13px] font-semibold">
              Current Team Stats
            </button>
          </Link>
          <Link to="/stats" className="btn-table">
            <button className="text-[13px] font-semibold">
              Current Odds Stats
            </button>
          </Link>
        </div>
      </div>
      <div
        className="dark-table ag-theme-alpine bg-dark w-[1100px]"
        style={{ height: 900, fontSize: 13 }}
      >
        <div className="options-select">
          <div className="options-select">
            <span
              className={`${
                typeState === 1 ? "selected-opt btn-options" : "btn-options"
              } border-l`}
            >
              <button
                onClick={handleClickPer100}
                className="text-[12px] font-semibold w-[87px]"
              >
                Per 100 Poss.
              </button>
            </span>
            <span
              className={`${
                typeState === 2 ? "selected-opt btn-options" : "btn-options"
              }`}
            >
              <button
                onClick={handleClickPerGame}
                className="text-[12px] font-semibold w-[87px]"
              >
                Per Game
              </button>
            </span>
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
          paginationPageSize={30}
          onCellClicked={cellClickedListener}
        />
      </div>
    </div>
  );
};

export default Table;
