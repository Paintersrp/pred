import { useEffect, useState } from "react";
import "./WIP.css";

function WIP() {
  const [data, setData] = useState([]);
  return (
    <div className="layout-form flex-wrap min-h-[500px]">
      <div className="text-center text-[64px]">WIP</div>
      <div className="outer13 flex flex-col justify-center text-center">
        <div className="cl13 flex w-[100px] h-[100px] text-black text-center justify-center items-center">
          I fucked your dad
        </div>
        HoverHere
      </div>
    </div>
  );
}

export default WIP;
