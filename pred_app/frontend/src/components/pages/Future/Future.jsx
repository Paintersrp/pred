import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import { AgGridReact } from "ag-grid-react";

import "ag-grid-community/styles/ag-grid.css";
import "./Future.css";
import "ag-grid-community/styles/ag-theme-alpine.css";

const Future = () => {
  const gridRef = useRef();
  const [rowData, setRowData] = useState();
  const [scoreData, setScoreData] = useState();

  const [columnDefs, setColumnDefs] = useState([
    { field: "Date", filter: true, minWidth: 125, flex: 1 },
    { field: "A_Team", minWidth: 200, flex: 1 },
    { field: "H_Team", minWidth: 200, flex: 1 },
    { field: "A_Odds", minWidth: 125, flex: 1 },
    { field: "H_Odds", minWidth: 125, flex: 1 },
  ]);

  const defaultColDef = useMemo(() => ({
    sortable: true,
    resizable: true,
  }));

  const cellClickedListener = useCallback((event) => {
    console.log("cellClicked", event);
  }, []);

  useEffect(() => {
    fetch("./data/upcoming.json")
      .then((result) => result.json())
      .then((rowData) => setRowData(rowData));
  }, []);

  return (
    <div className="layout-table text-white">
      <h2 className="set-width align-center text-xl p-2 mb-10 border set-bold">
        Future Predictions
      </h2>
      <div
        className="dark-table ag-theme-alpine bg-dark w-[837px]"
        style={{ height: 900, fontSize: 12 }}
      >
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
        <p className="set-semibold disclaim text-[12px] mb-1">
          *Predictions will change daily, adjusting for new data.
        </p>
      </div>
    </div>
  );
};

export default Future;
