theorem natDegree_C_le (a : R) : natDegree (C a) ≤ 0 := (natDegree_C a).le


theorem natDegree_natCast_le (n : ℕ) : natDegree (n : R[X]) ≤ 0 := (natDegree_natCast _).le

theorem natDegree_zero_le : natDegree (0 : R[X]) ≤ 0 := natDegree_zero.le

theorem natDegree_one_le : natDegree (1 : R[X]) ≤ 0 := natDegree_one.le


@[deprecated (since := "2024-04-17")]
alias natDegree_nat_cast_le := natDegree_natCast_le


theorem coeff_add_of_eq {n : ℕ} {a b : R} {f g : R[X]}
    (h_add_left : f.coeff n = a) (h_add_right : g.coeff n = b) :
                                  /-
                                    R : Type u_1
                                    inst✝ : Semiring R
                                    n : Nat
                                    a b : R
                                    f g : Polynomial R
                                    h_add_left : Eq (f.coeff n) a
                                    h_add_right : Eq (g.coeff n) b
                                    ⊢ Eq ((HAdd.hAdd f g).coeff n) (HAdd.hAdd a b)
                                  -/
    (f + g).coeff n = a + b := by subst ‹_› ‹_›; apply coeff_add
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem coeff_mul_add_of_le_natDegree_of_eq_ite {d df dg : ℕ} {a b : R} {f g : R[X]}
    (h_mul_left : natDegree f ≤ df) (h_mul_right : natDegree g ≤ dg)
    (h_mul_left : f.coeff df = a) (h_mul_right : g.coeff dg = b) (ddf : df + dg ≤ d) :
    (f * g).coeff d = if d = df + dg then a * b else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    d df dg : Nat
    a b : R
    f g : Polynomial R
    h_mul_left✝ : LE.le f.natDegree df
    h_mul_right✝ : LE.le g.natDegree dg
    h_mul_left : Eq (f.coeff df) a
    h_mul_right : Eq (g.coeff dg) b
    ddf : LE.le (HAdd.hAdd df dg) d
    ⊢ Eq ((HMul.hMul f g).coeff d) (ite (Eq d (HAdd.hAdd df dg)) (HMul.hMul a b) 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      d df dg : Nat
      a b : R
      f g : Polynomial R
      h_mul_left✝ : LE.le f.natDegree df
      h_mul_right✝ : LE.le g.natDegree dg
      h_mul_left : Eq (f.coeff df) a
      h_mul_right : Eq (g.coeff dg) b
      ddf : LE.le (HAdd.hAdd df dg) d
      h : Eq d (HAdd.hAdd df dg)
      ⊢ Eq ((HMul.hMul f g).coeff d) (HMul.hMul a b)
    -/
  · subst h_mul_left h_mul_right h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      df dg : Nat
      f g : Polynomial R
      h_mul_left : LE.le f.natDegree df
      h_mul_right : LE.le g.natDegree dg
      ddf : LE.le (HAdd.hAdd df dg) (HAdd.hAdd df dg)
      ⊢ Eq ((HMul.hMul f g).coeff (HAdd.hAdd df dg)) (HMul.hMul (f.coeff df) (g.coef …
    -/
    exact coeff_mul_of_natDegree_le ‹_› ‹_›
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      d df dg : Nat
      a b : R
      f g : Polynomial R
      h_mul_left✝ : LE.le f.natDegree df
      h_mul_right✝ : LE.le g.natDegree dg
      h_mul_left : Eq (f.coeff df) a
      h_mul_right : Eq (g.coeff dg) b
      ddf : LE.le (HAdd.hAdd df dg) d
      h : Not (Eq d (HAdd.hAdd df dg))
      ⊢ Eq ((HMul.hMul f g).coeff d) 0
    -/
  · apply coeff_eq_zero_of_natDegree_lt
    /-
      case neg.h
      R : Type u_1
      inst✝ : Semiring R
      d df dg : Nat
      a b : R
      f g : Polynomial R
      h_mul_left✝ : LE.le f.natDegree df
      h_mul_right✝ : LE.le g.natDegree dg
      h_mul_left : Eq (f.coeff df) a
      h_mul_right : Eq (g.coeff dg) b
      ddf : LE.le (HAdd.hAdd df dg) d
      h : Not (Eq d (HAdd.hAdd df dg))
      ⊢ LT.lt (HMul.hMul f g).natDegree d
    -/
    apply lt_of_le_of_lt ?_ (lt_of_le_of_ne ddf ?_)
      /-
        R : Type u_1
        inst✝ : Semiring R
        d df dg : Nat
        a b : R
        f g : Polynomial R
        h_mul_left✝ : LE.le f.natDegree df
        h_mul_right✝ : LE.le g.natDegree dg
        h_mul_left : Eq (f.coeff df) a
        h_mul_right : Eq (g.coeff dg) b
        ddf : LE.le (HAdd.hAdd df dg) d
        h : Not (Eq d (HAdd.hAdd df dg))
        ⊢ LE.le (HMul.hMul f g).natDegree (HAdd.hAdd df dg)
      -/
    · exact natDegree_mul_le_of_le ‹_› ‹_›
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        inst✝ : Semiring R
        d df dg : Nat
        a b : R
        f g : Polynomial R
        h_mul_left✝ : LE.le f.natDegree df
        h_mul_right✝ : LE.le g.natDegree dg
        h_mul_left : Eq (f.coeff df) a
        h_mul_right : Eq (g.coeff dg) b
        ddf : LE.le (HAdd.hAdd df dg) d
        h : Not (Eq d (HAdd.hAdd df dg))
        ⊢ Ne (HAdd.hAdd df dg) d
      -/
    · exact ne_comm.mp h
      /-
        🎉 no goals
      -/


theorem coeff_pow_of_natDegree_le_of_eq_ite' {m n o : ℕ} {a : R} {p : R[X]}
    (h_pow : natDegree p ≤ n) (h_exp : m * n ≤ o) (h_pow_bas : coeff p n = a) :
    coeff (p ^ m) o = if o = m * n then a ^ m else 0 := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    m n o : Nat
    a : R
    p : Polynomial R
    h_pow : LE.le p.natDegree n
    h_exp : LE.le (HMul.hMul m n) o
    h_pow_bas : Eq (p.coeff n) a
    ⊢ Eq ((HPow.hPow p m).coeff o) (ite (Eq o (HMul.hMul m n)) (HPow.hPow a m) 0)
  -/
  split_ifs with h
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      m n o : Nat
      a : R
      p : Polynomial R
      h_pow : LE.le p.natDegree n
      h_exp : LE.le (HMul.hMul m n) o
      h_pow_bas : Eq (p.coeff n) a
      h : Eq o (HMul.hMul m n)
      ⊢ Eq ((HPow.hPow p m).coeff o) (HPow.hPow a m)
    -/
  · subst h h_pow_bas
    /-
      case pos
      R : Type u_1
      inst✝ : Semiring R
      m n : Nat
      p : Polynomial R
      h_pow : LE.le p.natDegree n
      h_exp : LE.le (HMul.hMul m n) (HMul.hMul m n)
      ⊢ Eq ((HPow.hPow p m).coeff (HMul.hMul m n)) (HPow.hPow (p.coeff n) m)
    -/
    exact coeff_pow_of_natDegree_le ‹_›
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u_1
      inst✝ : Semiring R
      m n o : Nat
      a : R
      p : Polynomial R
      h_pow : LE.le p.natDegree n
      h_exp : LE.le (HMul.hMul m n) o
      h_pow_bas : Eq (p.coeff n) a
      h : Not (Eq o (HMul.hMul m n))
      ⊢ Eq ((HPow.hPow p m).coeff o) 0
    -/
  · apply coeff_eq_zero_of_natDegree_lt
    /-
      case neg.h
      R : Type u_1
      inst✝ : Semiring R
      m n o : Nat
      a : R
      p : Polynomial R
      h_pow : LE.le p.natDegree n
      h_exp : LE.le (HMul.hMul m n) o
      h_pow_bas : Eq (p.coeff n) a
      h : Not (Eq o (HMul.hMul m n))
      ⊢ LT.lt (HPow.hPow p m).natDegree o
    -/
    apply lt_of_le_of_lt ?_ (lt_of_le_of_ne ‹_› ?_)
      /-
        R : Type u_1
        inst✝ : Semiring R
        m n o : Nat
        a : R
        p : Polynomial R
        h_pow : LE.le p.natDegree n
        h_exp : LE.le (HMul.hMul m n) o
        h_pow_bas : Eq (p.coeff n) a
        h : Not (Eq o (HMul.hMul m n))
        ⊢ LE.le (HPow.hPow p m).natDegree (HMul.hMul m n)
      -/
    · exact natDegree_pow_le_of_le m ‹_›
      /-
        🎉 no goals
      -/
      /-
        R : Type u_1
        inst✝ : Semiring R
        m n o : Nat
        a : R
        p : Polynomial R
        h_pow : LE.le p.natDegree n
        h_exp : LE.le (HMul.hMul m n) o
        h_pow_bas : Eq (p.coeff n) a
        h : Not (Eq o (HMul.hMul m n))
        ⊢ Ne (HMul.hMul m n) o
      -/
    · exact Iff.mp ne_comm h
      /-
        🎉 no goals
      -/


theorem natDegree_smul_le_of_le {n : ℕ} {a : R} {f : R[X]} (hf : natDegree f ≤ n) :
    natDegree (a • f) ≤ n :=
  (natDegree_smul_le a f).trans hf


theorem degree_smul_le_of_le {n : ℕ} {a : R} {f : R[X]} (hf : degree f ≤ n) :
    degree (a • f) ≤ n :=
  (degree_smul_le a f).trans hf


theorem coeff_smul {n : ℕ} {a : R} {f : R[X]} : (a • f).coeff n = a * f.coeff n := rfl


/--  The following two lemmas should be viewed as a hand-made "congr"-lemmas.
They achieve the following goals.
* They introduce *two* fresh metavariables replacing the given one `deg`,
  one for the `natDegree ≤` computation and one for the `coeff =` computation.
  This helps `compute_degree`, since it does not "pre-estimate" the degree,
  but it "picks it up along the way".
* They split checking the inequality `coeff p n ≠ 0` into the task of
  finding a value `c` for the `coeff` and then
  proving that this value is non-zero by `coeff_ne_zero`.
-/
theorem natDegree_eq_of_le_of_coeff_ne_zero' {deg m o : ℕ} {c : R} {p : R[X]}
    (h_natDeg_le : natDegree p ≤ m) (coeff_eq : coeff p o = c)
    (coeff_ne_zero : c ≠ 0) (deg_eq_deg : m = deg) (coeff_eq_deg : o = deg) :
    natDegree p = deg := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    deg m o : Nat
    c : R
    p : Polynomial R
    h_natDeg_le : LE.le p.natDegree m
    coeff_eq : Eq (p.coeff o) c
    coeff_ne_zero : Ne c 0
    deg_eq_deg : Eq m deg
    coeff_eq_deg : Eq o deg
    ⊢ Eq p.natDegree deg
  -/
  subst coeff_eq deg_eq_deg coeff_eq_deg
  /-
    R : Type u_1
    inst✝ : Semiring R
    o : Nat
    p : Polynomial R
    coeff_ne_zero : Ne (p.coeff o) 0
    h_natDeg_le : LE.le p.natDegree o
    ⊢ Eq p.natDegree o
  -/
  exact natDegree_eq_of_le_of_coeff_ne_zero ‹_› ‹_›
  /-
    🎉 no goals
  -/


theorem degree_eq_of_le_of_coeff_ne_zero' {deg m o : WithBot ℕ} {c : R} {p : R[X]}
    (h_deg_le : degree p ≤ m) (coeff_eq : coeff p (WithBot.unbot' 0 deg) = c)
    (coeff_ne_zero : c ≠ 0) (deg_eq_deg : m = deg) (coeff_eq_deg : o = deg) :
    degree p = deg := by
  /-
    R : Type u_1
    inst✝ : Semiring R
    deg m o : WithBot Nat
    c : R
    p : Polynomial R
    h_deg_le : LE.le p.degree m
    coeff_eq : Eq (p.coeff (WithBot.unbot' 0 deg)) c
    coeff_ne_zero : Ne c 0
    deg_eq_deg : Eq m deg
    coeff_eq_deg : Eq o deg
    ⊢ Eq p.degree deg
  -/
  subst coeff_eq coeff_eq_deg deg_eq_deg
  /-
    R : Type u_1
    inst✝ : Semiring R
    m : WithBot Nat
    p : Polynomial R
    h_deg_le : LE.le p.degree m
    coeff_ne_zero : Ne (p.coeff (WithBot.unbot' 0 m)) 0
    ⊢ Eq p.degree m
  -/
  rcases eq_or_ne m ⊥ with rfl|hh
    /-
      case inl
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      h_deg_le : LE.le p.degree Bot.bot
      coeff_ne_zero : Ne (p.coeff (WithBot.unbot' 0 Bot.bot)) 0
      ⊢ Eq p.degree Bot.bot
    -/
  · exact bot_unique h_deg_le
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u_1
      inst✝ : Semiring R
      m : WithBot Nat
      p : Polynomial R
      h_deg_le : LE.le p.degree m
      coeff_ne_zero : Ne (p.coeff (WithBot.unbot' 0 m)) 0
      hh : Ne m Bot.bot
      ⊢ Eq p.degree m
    -/
  · obtain ⟨m, rfl⟩ := WithBot.ne_bot_iff_exists.mp hh
    /-
      case inr.intro
      R : Type u_1
      inst✝ : Semiring R
      p : Polynomial R
      m : Nat
      h_deg_le : LE.le p.degree ↑m
      coeff_ne_zero : Ne (p.coeff (WithBot.unbot' 0 ↑m)) 0
      hh : Ne (↑m) Bot.bot
      ⊢ Eq p.degree ↑m
    -/
    exact degree_eq_of_le_of_coeff_ne_zero ‹_› ‹_›
    /-
      🎉 no goals
    -/


theorem coeff_congr_lhs (h : coeff f m = r) (natDeg_eq_coeff : m = n) : coeff f n = r :=
  natDeg_eq_coeff ▸ h

theorem coeff_congr (h : coeff f m = r) (natDeg_eq_coeff : m = n) {s : R} (rs : r = s) :
    coeff f n = s :=
  natDeg_eq_coeff ▸ rs ▸ h


theorem natDegree_intCast_le (n : ℤ) : natDegree (n : R[X]) ≤ 0 := (natDegree_intCast _).le


@[deprecated (since := "2024-04-17")]
alias natDegree_int_cast_le := natDegree_intCast_le


theorem coeff_sub_of_eq {n : ℕ} {a b : R} {f g : R[X]} (hf : f.coeff n = a) (hg : g.coeff n = b) :
                                  /-
                                    R : Type u_1
                                    inst✝ : Ring R
                                    n : Nat
                                    a b : R
                                    f g : Polynomial R
                                    hf : Eq (f.coeff n) a
                                    hg : Eq (g.coeff n) b
                                    ⊢ Eq ((HSub.hSub f g).coeff n) (HSub.hSub a b)
                                  -/
    (f - g).coeff n = a - b := by subst hf hg; apply coeff_sub
                                               /-
                                                 🎉 no goals
                                               -/


theorem coeff_intCast_ite {n : ℕ} {a : ℤ} : (Int.cast a : R[X]).coeff n = ite (n = 0) a 0 := by
  /-
    R : Type u_1
    inst✝ : Ring R
    n : Nat
    a : Int
    ⊢ Eq ((↑a).coeff n) ↑(ite (Eq n 0) a 0)
  -/
  simp only [← C_eq_intCast, coeff_C, Int.cast_ite, Int.cast_zero]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias coeff_int_cast_ite := coeff_intCast_ite


/-- `twoHeadsArgs e` takes an `Expr`ession `e` as input and recurses into `e` to make sure
the `e` looks like `lhs ≤ rhs` or `lhs = rhs` and that `lhs` is one of
`natDegree f, degree f, coeff f d`.
It returns
* the function being applied on the LHS (`natDegree`, `degree`, or `coeff`),
  or else `.anonymous` if it's none of these.
* the name of the relation (`Eq` or `LE.le`), or else `.anonymous` if it's none of these.
* either
  * `.inl zero`, `.inl one`, or `.inl many` if the polynomial in a numeral
  * or `.inr` of the head symbol of `f`
  * or `.inl .anonymous` if inapplicable
* if it exists, whether the `rhs` is a metavariable
* if the LHS is `coeff f d`, whether `d` is a metavariable

This is all the data needed to figure out whether `compute_degree` can make progress on `e`
and, if so, which lemma it should apply.

Sample outputs:
* `natDegree (f + g) ≤ d => (natDegree, LE.le, HAdd.hAdd, d.isMVar, none)` (similarly for `=`);
* `degree (f * g) = d => (degree, Eq, HMul.hMul, d.isMVar, none)` (similarly for `≤`);
* `coeff (1 : ℕ[X]) c = x => (coeff, Eq, one, x.isMVar, c.isMVar)` (no `≤` option!).
-/
def twoHeadsArgs (e : Expr) : Name × Name × (Name ⊕ Name) × List Bool := Id.run do
  let (eq_or_le, lhs, rhs) ← match e.getAppFnArgs with
    | (na@``Eq, #[_, lhs, rhs])       => pure (na, lhs, rhs)
    | (na@``LE.le, #[_, _, lhs, rhs]) => pure (na, lhs, rhs)
    | _ => return (.anonymous, .anonymous, .inl .anonymous, [])
  let (ndeg_or_deg_or_coeff, pol, and?) ← match lhs.getAppFnArgs with
    | (na@``Polynomial.natDegree, #[_, _, pol])     => (na, pol, [rhs.isMVar])
    | (na@``Polynomial.degree,    #[_, _, pol])     => (na, pol, [rhs.isMVar])
    | (na@``Polynomial.coeff,     #[_, _, pol, c])  => (na, pol, [rhs.isMVar, c.isMVar])
    | _ => return (.anonymous, eq_or_le, .inl .anonymous, [])
  let head := match pol.numeral? with
    -- can I avoid the tri-splitting `n = 0`, `n = 1`, and generic `n`?
    | some 0 => .inl `zero
    | some 1 => .inl `one
    | some _ => .inl `many
    | none => match pol.getAppFnArgs with
      | (``DFunLike.coe, #[_, _, _, _, polFun, _]) =>
        let na := polFun.getAppFn.constName
        if na ∈ [``Polynomial.monomial, ``Polynomial.C] then
          .inr na
        else
          .inl .anonymous
      | (na, _) => .inr na
  (ndeg_or_deg_or_coeff, eq_or_le, head, and?)


/--
`getCongrLemma (lhs_name, rel_name, Mvars?)` returns the name of a lemma that preprocesses
one of the five target
*  `natDegree f ≤ d`;
*  `natDegree f = d`.
*  `degree f ≤ d`;
*  `degree f = d`.
*  `coeff f d = r`.

The end goals are of the form
* `natDegree f ≤ ?_`, `degree f ≤ ?_`, `coeff f ?_ = ?_`, with fresh metavariables;
* `coeff f m ≠ s` with `m, s` not necessarily metavariables;
* several equalities/inequalities between expressions and assignments for metavariables.

`getCongrLemma` gets called at the very beginning of `compute_degree` and whenever an intermediate
goal does not have the right metavariables.
Note that the side-goals of the congruence lemma are neither of the form `natDegree f = d` nor
of the form `degree f = d`.

`getCongrLemma` admits an optional "debug" flag: `getCongrLemma data true` prints the name of
the congruence lemma that it returns.
-/
def getCongrLemma (twoH : Name × Name × List Bool) (debug : Bool := false) : Name :=
  let nam := match twoH with
    | (_,           ``LE.le, [rhs]) => if rhs then ``id else ``le_trans
    | (``natDegree, ``Eq, [rhs])    => if rhs then ``id else ``natDegree_eq_of_le_of_coeff_ne_zero'
    | (``degree,    ``Eq, [rhs])    => if rhs then ``id else ``degree_eq_of_le_of_coeff_ne_zero'
    | (``coeff,     ``Eq, [rhs, c]) =>
      match rhs, c with
      | false, false => ``coeff_congr
      | false, true  => ``Eq.trans
      | true, false  => ``coeff_congr_lhs
      | true, true   => ``id
    | _ => ``id
  if debug then
    let last := nam.lastComponentAsString
    let natr := if last == "trans" then nam.toString else last
    dbg_trace f!"congr lemma: '{natr}'"
    nam
  else
    nam


/--
`dispatchLemma twoH` takes its input `twoH` from the output of `twoHeadsArgs`.

Using the information contained in `twoH`, it decides which lemma is the most appropriate.

`dispatchLemma` is essentially the main dictionary for `compute_degree`.
-/
--  Internally, `dispatchLemma` produces 3 names: these are the lemmas that are appropriate
--  for goals of the form `natDegree f ≤ d`, `degree f ≤ d`, `coeff f d = a`, in this order.
def dispatchLemma
    (twoH : Name × Name × (Name ⊕ Name) × List Bool) (debug : Bool := false) : Name :=
  match twoH with
    | (.anonymous, _, _) => ``id -- `twoH` gave default value, so we do nothing
    | (_, .anonymous, _) => ``id -- `twoH` gave default value, so we do nothing
    | (na1, na2, head, bools) =>
      let msg := f!"\ndispatchLemma:\n  {head}"
      -- if there is some non-metavariable on the way, we "congr" it away
      if false ∈ bools then getCongrLemma (na1, na2, bools) debug
      else
      -- otherwise, we select either the first, second or third element of the triple in `nas` below
      let π (natDegLE : Name) (degLE : Name) (coeff : Name) : Name := Id.run do
        let lem := match na1, na2 with
          | ``natDegree, ``LE.le => natDegLE
          | ``degree, ``LE.le => degLE
          | ``coeff, ``Eq => coeff
          | _, ``LE.le => ``le_rfl
          | _, _ => ``rfl
        if debug then
          dbg_trace f!"{lem.lastComponentAsString}\n{msg}"
        lem
      match head with
        | .inl `zero => π ``natDegree_zero_le ``degree_zero_le ``coeff_zero
        | .inl `one  => π ``natDegree_one_le ``degree_one_le ``coeff_one
        | .inl `many => π ``natDegree_natCast_le ``degree_natCast_le ``coeff_natCast_ite
        | .inl .anonymous => π ``le_rfl ``le_rfl ``rfl
        | .inr ``HAdd.hAdd =>
          π ``natDegree_add_le_of_le ``degree_add_le_of_le ``coeff_add_of_eq
        | .inr ``HSub.hSub =>
          π ``natDegree_sub_le_of_le ``degree_sub_le_of_le ``coeff_sub_of_eq
        | .inr ``HMul.hMul =>
          π ``natDegree_mul_le_of_le ``degree_mul_le_of_le ``coeff_mul_add_of_le_natDegree_of_eq_ite
        | .inr ``HPow.hPow =>
          π ``natDegree_pow_le_of_le ``degree_pow_le_of_le ``coeff_pow_of_natDegree_le_of_eq_ite'
        | .inr ``Neg.neg =>
          π ``natDegree_neg_le_of_le ``degree_neg_le_of_le ``coeff_neg
        | .inr ``Polynomial.X =>
          π ``natDegree_X_le ``degree_X_le ``coeff_X
        | .inr ``Nat.cast =>
          π ``natDegree_natCast_le ``degree_natCast_le ``coeff_natCast_ite
        | .inr ``NatCast.natCast =>
          π ``natDegree_natCast_le ``degree_natCast_le ``coeff_natCast_ite
        | .inr ``Int.cast =>
          π ``natDegree_intCast_le ``degree_intCast_le ``coeff_intCast_ite
        | .inr ``IntCast.intCast =>
          π ``natDegree_intCast_le ``degree_intCast_le ``coeff_intCast_ite
        | .inr ``Polynomial.monomial =>
          π ``natDegree_monomial_le ``degree_monomial_le ``coeff_monomial
        | .inr ``Polynomial.C =>
          π ``natDegree_C_le ``degree_C_le ``coeff_C
        | .inr ``HSMul.hSMul =>
          π ``natDegree_smul_le_of_le ``degree_smul_le_of_le ``coeff_smul
        | _ => π ``le_rfl ``le_rfl ``rfl


/-- `try_rfl mvs` takes as input a list of `MVarId`s, scans them partitioning them into two
lists: the goals containing some metavariables and the goals not containing any metavariable.

If a goal containing a metavariable has the form `?_ = x`, `x = ?_`, where `?_` is a metavariable
and `x` is an expression that does not involve metavariables, then it closes this goal using `rfl`,
effectively assigning the metavariable to `x`.

If a goal does not contain metavariables, it tries `rfl` on it.

It returns the list of `MVarId`s, beginning with the ones that initially involved (`Expr`)
metavariables followed by the rest.
-/
def try_rfl (mvs : List MVarId) : MetaM (List MVarId) := do
  let (yesMV, noMV) := ← mvs.partitionM fun mv =>
                          return hasExprMVar (← instantiateMVars (← mv.getDecl).type)
  let tried_rfl := ← noMV.mapM fun g => g.applyConst ``rfl <|> return [g]
  let assignable := ← yesMV.mapM fun g => do
    let tgt := ← instantiateMVars (← g.getDecl).type
    match tgt.eq? with
      | some (_, lhs, rhs) =>
        if (isMVar rhs && (! hasExprMVar lhs)) ||
           (isMVar lhs && (! hasExprMVar rhs)) then
           g.applyConst ``rfl
        else pure [g]
      | none =>
        return [g]
  return (assignable.flatten ++ tried_rfl.flatten)


/--
`splitApply mvs static` takes two lists of `MVarId`s.  The first list, `mvs`,
corresponds to goals that are potentially within the scope of `compute_degree`:
namely, goals of the form
`natDegree f ≤ d`, `degree f ≤ d`, `natDegree f = d`, `degree f = d`, `coeff f d = r`.

`splitApply` determines which of these goals are actually within the scope, it applies the relevant
lemma and returns two lists: the left-over goals of all the applications, followed by the
concatenation of the previous `static` list, followed by the newly discovered goals outside of the
scope of `compute_degree`. -/
def splitApply (mvs static : List MVarId) : MetaM ((List MVarId) × (List MVarId)) := do
  let (can_progress, curr_static) := ← mvs.partitionM fun mv => do
    return dispatchLemma (twoHeadsArgs (← mv.getType'')) != ``id
  let progress := ← can_progress.mapM fun mv => do
    let lem := dispatchLemma <| twoHeadsArgs (← mv.getType'')
    mv.applyConst <| lem
  return (progress.flatten, static ++ curr_static)


/-- `miscomputedDegree? deg false_goals` takes as input
*  an `Expr`ession `deg`, representing the degree of a polynomial
   (i.e. an `Expr`ession of inferred type either `ℕ` or `WithBot ℕ`);
*  a list of `MVarId`s `false_goals`.

Although inconsequential for this function, the list of goals `false_goals` reduces to `False`
if `norm_num`med.
`miscomputedDegree?` extracts error information from goals of the form
*  `a ≠ b`, assuming it comes from `⊢ coeff_of_given_degree ≠ 0`
   -- reducing to `False` means that the coefficient that was supposed to vanish, does not;
*  `a ≤ b`, assuming it comes from `⊢ degree_of_subterm ≤ degree_of_polynomial`
   -- reducing to `False` means that there is a term of degree that is apparently too large;
*  `a = b`, assuming it comes from `⊢ computed_degree ≤ given_degree`
   -- reducing to `False` means that there is a term of degree that is apparently too large.

The cases `a ≠ b` and `a = b` are not a perfect match with the top coefficient:
reducing to `False` is not exactly correlated with a coefficient being non-zero.
It does mean that `compute_degree` reduced the initial goal to an unprovable state
(unless there was already a contradiction in the initial hypotheses!), but it is indicative that
there may be some problem.
-/
def miscomputedDegree? (deg : Expr) : List Expr → List MessageData
  | tgt::tgts =>
    let rest := miscomputedDegree? deg tgts
    if tgt.ne?.isSome then
      m!"* the coefficient of degree {deg} may be zero" :: rest
    else if let some ((Expr.const ``Nat []), lhs, _) := tgt.le? then
      m!"* there is at least one term of naïve degree {lhs}" :: rest
    else if let some (_, lhs, _) := tgt.eq? then
      m!"* there may be a term of naïve degree {lhs}" :: rest
    else rest
  | [] => []


/--
`compute_degree` is a tactic to solve goals of the form
*  `natDegree f = d`,
*  `degree f = d`,
*  `natDegree f ≤ d`,
*  `degree f ≤ d`,
*  `coeff f d = r`, if `d` is the degree of `f`.

The tactic may leave goals of the form `d' = d` `d' ≤ d`, or `r ≠ 0`, where `d'` in `ℕ` or
`WithBot ℕ` is the tactic's guess of the degree, and `r` is the coefficient's guess of the
leading coefficient of `f`.

`compute_degree` applies `norm_num` to the left-hand side of all side goals, trying to close them.

The variant `compute_degree!` first applies `compute_degree`.
Then it uses `norm_num` on all the whole remaining goals and tries `assumption`.
-/
syntax (name := computeDegree) "compute_degree" "!"? : tactic


           /-
             ⊢ Lean.Name
           -/
initialize registerTraceClass `Tactic.compute_degree
           /-
             🎉 no goals
           -/


@[inherit_doc computeDegree]
macro "compute_degree!" : tactic => `(tactic| compute_degree !)


elab_rules : tactic | `(tactic| compute_degree $[!%$bang]?) => focus <| withMainContext do
  let goal ← getMainGoal
  let gt ← goal.getType''
  let deg? := match gt.eq? with
    | some (_, _, rhs) => some rhs
    | _ => none
  let twoH := twoHeadsArgs gt
  match twoH with
    | (_, .anonymous, _) => throwError m!"'compute_degree' inapplicable. \
        The goal{indentD gt}\nis expected to be '≤' or '='."
    | (.anonymous, _, _) => throwError m!"'compute_degree' inapplicable. \
        The LHS must be an application of 'natDegree', 'degree', or 'coeff'."
    | _ =>
      let lem := dispatchLemma twoH
      trace[Tactic.compute_degree]
        f!"'compute_degree' first applies lemma '{lem.lastComponentAsString}'"
      let mut (gls, static) := (← goal.applyConst lem, [])
      while gls != [] do (gls, static) ← splitApply gls static
      let rfled ← try_rfl static
      setGoals rfled
      --  simplify the left-hand sides, since this is where the degree computations leave
      --  expressions such as `max (0 * 1) (max (1 + 0 + 3 * 4) (7 * 0))`
      evalTactic
        (← `(tactic| try any_goals conv_lhs =>
                       (simp +decide only [Nat.cast_withBot]; norm_num)))
      if bang.isSome then
        let mut false_goals : Array MVarId := #[]
        let mut new_goals : Array MVarId := #[]
        for g in ← getGoals do
          let gs' ← run g do evalTactic (←
            `(tactic| try (any_goals norm_num <;> norm_cast <;> try assumption)))
          new_goals := new_goals ++ gs'.toArray
          if ← gs'.anyM fun g' => g'.withContext do return (← g'.getType'').isConstOf ``False then
            false_goals := false_goals.push g
        setGoals new_goals.toList
        if let some deg := deg? then
          let errors := miscomputedDegree? deg (← false_goals.mapM (MVarId.getType'' ·)).toList
          unless errors.isEmpty do
            throwError Lean.MessageData.joinSep
              (m!"The given degree is '{deg}'.  However,\n" :: errors) "\n"


/-- `monicity` tries to solve a goal of the form `Monic f`.
It converts the goal into a goal of the form `natDegree f ≤ n` and one of the form `f.coeff n = 1`
and calls `compute_degree` on those two goals.

The variant `monicity!` starts like `monicity`, but calls `compute_degree!` on the two side-goals.
-/
macro (name := monicityMacro) "monicity" : tactic =>
  `(tactic| (apply monic_of_natDegree_le_of_coeff_eq_one <;> compute_degree))


@[inherit_doc monicityMacro]
macro "monicity!" : tactic =>
  `(tactic| (apply monic_of_natDegree_le_of_coeff_eq_one <;> compute_degree!))


