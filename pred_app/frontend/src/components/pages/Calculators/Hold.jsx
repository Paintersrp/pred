import { useState } from "react";
import axios from "axios";
import { Link } from "react-router-dom";
import "./Calculators.css";

function Hold() {
  const [toggleState, setToggleState] = useState(1);
  const [typeState, setTypeState] = useState(1);

  const [favNum, setFav] = useState();
  const [underNum, setUnder] = useState();

  const [favImplied, setFavImplied] = useState(0);
  const [underImplied, setUnderImplied] = useState(0);
  const [holdNum, setHoldNum] = useState(0);

  const toggleOption = (index) => {
    setToggleState(index);
  };

  const toggleType = (index) => {
    setTypeState(index);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const calcNums = { favNum, underNum, typeState };

    const data = await axios
      .post("http://54.161.55.120:8000/api/hold/", calcNums)
      .then((res) => {
        console.log(res.data);
        setFavImplied(res.data.fav_implied);
        setUnderImplied(res.data.under_implied);
        setHoldNum(res.data.hold);
      });
  };

  return (
    <div className="layout flex-wrap">
      <div className="setwidth content-center border-r border-l">
        <div className="calc-select flex flex-row">
          <Link to="/calc/payout" className="btn-link calc-btn-select">
            <button className="">Payout</button>
          </Link>
          <Link to="/calc/hedge" className="btn-link calc-btn-select">
            <button className="">Hedge</button>
          </Link>
          <Link to="/calc/hold" className="btn-link">
            <button className="selected">Hold</button>
          </Link>
        </div>
        <div
          className={
            toggleState === 1
              ? "content  active-content CalcTabs"
              : "content CalcTabs"
          }
        >
          <h2 className="mt-6 mb-10 title">Hold Calculator</h2>

          <div className="mb-10">
            <div className="align-center mb-2">Odds Input Type</div>
            <hr className="mb-4" />
            <div className="flex flex-row">
              <div
                className={`${
                  typeState === 1
                    ? "flex-1 btn-link"
                    : "flex-1 btn-link calc-btn-select"
                }`}
              >
                <button
                  className={`${
                    typeState === 1 ? "selected flex-1" : "flex-1"
                  }`}
                  onClick={() => toggleType(1)}
                >
                  American
                </button>
              </div>
              <div
                className={`${
                  typeState === 2
                    ? "flex-1 btn-link"
                    : "flex-1 btn-link calc-btn-select"
                }`}
              >
                <button
                  className={`${
                    typeState === 2 ? "selected flex-1" : "flex-1"
                  }`}
                  onClick={() => toggleType(2)}
                >
                  Decimal
                </button>
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="form mb-12">
            <label className="">Odds #1</label>
            <input
              type="number"
              step="0.01"
              placeholder={typeState === 2 ? "1.5" : "900"}
              required
              name="favNum"
              value={favNum}
              onChange={(e) => setFav(e.target.value)}
            />

            <label className="">Odds #2</label>
            <input
              type="number"
              placeholder={typeState === 2 ? "1.5" : "900"}
              required
              name="underNum"
              value={underNum}
              onChange={(e) => setUnder(e.target.value)}
            />

            <button className="w-[25%] mt-6 calc-btn font-semibold">
              Calculate
            </button>
          </form>

          <div className="div mb-10">
            <h2 className="align-center mb-2">Hold Results</h2>
            <hr className="mb-4" />
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center text-[14px]">
                #1 Implied
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                #2 Implied
              </label>
              <label className="flex-1 flex-row align-center text-[14px]">
                Bookie Hold
              </label>
            </div>
            <div className="flex flex-row align-center">
              <label className="flex-1 flex-row align-center border text-[14px]">
                {favImplied}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {underImplied}
              </label>
              <label className="flex-1 flex-row align-center border text-[14px]">
                {holdNum}
              </label>
            </div>
          </div>
        </div>
      </div>
      <div className="setwidth-info border-r border-l flex flex-col flex-wrap info-sm">
        <div className="calc-select flex flex-row h-[71px] justify-center items-center text-white">
          <h2 className="text-[18px] font-semibold">Information</h2>
        </div>
        <div className="active-content flex">
          <h2 className="text-white p-2 underline underline-offset-4">
            What is a Bookie Hold?
          </h2>
          <p className="text-white p-2 text-[13px]">
            A Bookmaker's Hold refers to the percentage the Bookmaker keeps for
            every dollar wagered. For most Bookmakers, this is around 4%.
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What are American Odds?
          </h2>
          <p className="text-white p-2 text-[13px]">
            American Odds are commonly used amongst American bookmakers, as the
            name suggests, and center around a baseline value of $100.
          </p>
          <p className="text-white p-2 text-[13px]">
            When looking at the odds of the favorited team, you might see -145.
            This means that you must risk $145 in order to make $100.
          </p>
          <p className="text-white p-2 text-[13px]">
            For the underdog, you might see +145 (or just 145). This means that
            you will make $145 for a $100 Wager
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What are Decimal Odds?
          </h2>
          <p className="text-white p-2 text-[13px]">
            Decimal Odds are another commonly used system by Bookmakers. With
            decimal odds, odds are presented in a decimal format such as 1.41.
          </p>
          <p className="text-white p-2 text-[13px]">
            Calculating the payout of a wager using decimal odds is fairly
            straightforward. If you wager $100 on Boston at a decimal odds of
            1.41, your return would be $141.
          </p>
          <h2 className="text-white p-2 underline underline-offset-4">
            What are Implied Odds?
          </h2>
          <p className="text-white p-2 text-[13px]">
            Implied Odds are determine based on a Bookmaker's given Decimal or
            American Odds. They represent the likelihood of a team winning, on a
            1-100% scale.
          </p>
          <p className="text-white p-2 text-[13px]">
            For example, Boston has -200 American Odds in their upcoming game vs
            Washington. The Implied Odds of Boston winning the game (at -200)
            are 66.66%.
          </p>
          <p className="text-white p-2 text-[13px]">
            The formula for determining Implied Odds is: (1 / Converted Decimal
            Odds) * 100. (American Odds are Converted to Decimal)
          </p>
        </div>
      </div>
    </div>
  );
}

export default Hold;
