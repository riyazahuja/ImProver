/-- A separable contraction of a polynomial `f` is a separable polynomial `g` such that
`g(x^(q^m)) = f(x)` for some `m : ℕ`. -/
def IsSeparableContraction (f : F[X]) (g : F[X]) : Prop :=
  g.Separable ∧ ∃ m : ℕ, expand F (q ^ m) g = f


/-- The condition of having a separable contraction. -/
def HasSeparableContraction (f : F[X]) : Prop :=
  ∃ g : F[X], IsSeparableContraction q f g


/-- A choice of a separable contraction. -/
def HasSeparableContraction.contraction : F[X] :=
  Classical.choose hf


/-- The separable degree of a polynomial is the degree of a given separable contraction. -/
def HasSeparableContraction.degree : ℕ :=
  hf.contraction.natDegree


/-- The `HasSeparableContraction.contraction` is indeed a separable contraction. -/
theorem HasSeparableContraction.isSeparableContraction :
    IsSeparableContraction q f hf.contraction := Classical.choose_spec hf


/-- The separable degree divides the degree, in function of the exponential characteristic of F. -/
theorem IsSeparableContraction.dvd_degree' {g} (hf : IsSeparableContraction q f g) :
    ∃ m : ℕ, g.natDegree * q ^ m = f.natDegree := by
  /-
    F : Type u_1
    inst✝ : CommSemiring F
    q : Nat
    f g : Polynomial F
    hf : Polynomial.IsSeparableContraction q f g
    ⊢ Exists fun m => Eq (HMul.hMul g.natDegree (HPow.hPow q m)) f.natDegree
  -/
  obtain ⟨m, rfl⟩ := hf.2
  /-
    case intro
    F : Type u_1
    inst✝ : CommSemiring F
    q : Nat
    g : Polynomial F
    m : Nat
    hf : Polynomial.IsSeparableContraction q ((Polynomial.expand F (HPow.hPow q m) …
    ⊢ Exists fun m_1 => Eq (HMul.hMul g.natDegree (HPow.hPow q m_1)) ((Polynomial. …
  -/
  use m
  /-
    case h
    F : Type u_1
    inst✝ : CommSemiring F
    q : Nat
    g : Polynomial F
    m : Nat
    hf : Polynomial.IsSeparableContraction q ((Polynomial.expand F (HPow.hPow q m) …
    ⊢ Eq (HMul.hMul g.natDegree (HPow.hPow q m)) ((Polynomial.expand F (HPow.hPow  …
  -/
  rw [natDegree_expand]
  /-
    🎉 no goals
  -/


theorem HasSeparableContraction.dvd_degree' : ∃ m : ℕ, hf.degree * q ^ m = f.natDegree :=
  (Classical.choose_spec hf).dvd_degree'


/-- The separable degree divides the degree. -/
theorem HasSeparableContraction.dvd_degree : hf.degree ∣ f.natDegree :=
  let ⟨a, ha⟩ := hf.dvd_degree'
  Dvd.intro (q ^ a) ha


/-- In exponential characteristic one, the separable degree equals the degree. -/
theorem HasSeparableContraction.eq_degree {f : F[X]} (hf : HasSeparableContraction 1 f) :
    hf.degree = f.natDegree := by
  /-
    F : Type u_1
    inst✝ : CommSemiring F
    f : Polynomial F
    hf : Polynomial.HasSeparableContraction 1 f
    ⊢ Eq hf.degree f.natDegree
  -/
  let ⟨a, ha⟩ := hf.dvd_degree'
  /-
    F : Type u_1
    inst✝ : CommSemiring F
    f : Polynomial F
    hf : Polynomial.HasSeparableContraction 1 f
    a : Nat
    ha : Eq (HMul.hMul hf.degree (HPow.hPow 1 a)) f.natDegree
    ⊢ Eq hf.degree f.natDegree
  -/
  rw [← ha, one_pow a, mul_one]
  /-
    🎉 no goals
  -/


/-- Every irreducible polynomial can be contracted to a separable polynomial. -/
@[stacks 09H0]
theorem _root_.Irreducible.hasSeparableContraction (q : ℕ) [hF : ExpChar F q] {f : F[X]}
    (irred : Irreducible f) : HasSeparableContraction q f := by
  /-
    F : Type u_1
    inst✝ : Field F
    q : Nat
    hF : ExpChar F q
    f : Polynomial F
    irred : Irreducible f
    ⊢ Polynomial.HasSeparableContraction q f
  -/
  cases hF
    /-
      case zero
      F : Type u_1
      inst✝¹ : Field F
      f : Polynomial F
      irred : Irreducible f
      inst✝ : CharZero F
      ⊢ Polynomial.HasSeparableContraction 1 f
    -/
  · exact ⟨f, irred.separable, ⟨0, by rw [pow_zero, expand_one]⟩⟩
    /-
      🎉 no goals
    -/
    /-
      case prime
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      irred : Irreducible f
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      ⊢ Polynomial.HasSeparableContraction q f
    -/
  · rcases exists_separable_of_irreducible q irred ‹q.Prime›.ne_zero with ⟨n, g, hgs, hge⟩
    /-
      case prime.intro.intro.intro
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      irred : Irreducible f
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      n : Nat
      g : Polynomial F
      hgs : g.Separable
      hge : Eq ((Polynomial.expand F (HPow.hPow q n)) g) f
      ⊢ Polynomial.HasSeparableContraction q f
    -/
    exact ⟨g, hgs, n, hge⟩
    /-
      🎉 no goals
    -/


/-- If two expansions (along the positive characteristic) of two separable polynomials `g` and `g'`
agree, then they have the same degree. -/
theorem contraction_degree_eq_or_insep [hq : NeZero q] [CharP F q] (g g' : F[X]) (m m' : ℕ)
    (h_expand : expand F (q ^ m) g = expand F (q ^ m') g') (hg : g.Separable) (hg' : g'.Separable) :
    g.natDegree = g'.natDegree := by
  /-
    F : Type u_1
    inst✝¹ : Field F
    q : Nat
    hq : NeZero q
    inst✝ : CharP F q
    g g' : Polynomial F
    m m' : Nat
    h_expand : Eq ((Polynomial.expand F (HPow.hPow q m)) g) ((Polynomial.expand F  …
    hg : g.Separable
    hg' : g'.Separable
    ⊢ Eq g.natDegree g'.natDegree
  -/
  wlog hm : m ≤ m'
    /-
      case inr
      F : Type u_1
      inst✝¹ : Field F
      q : Nat
      hq : NeZero q
      inst✝ : CharP F q
      g g' : Polynomial F
      m m' : Nat
      h_expand : Eq ((Polynomial.expand F (HPow.hPow q m)) g) ((Polynomial.expand F  …
      hg : g.Separable
      hg' : g'.Separable
      this : ∀ {F : Type u_1} [inst : Field F] (q : Nat) [hq : NeZero q] [inst_1 : C …
      hm : Not (LE.le m m')
      ⊢ Eq g.natDegree g'.natDegree
    -/
  · exact (this q g' g m' m h_expand.symm hg' hg (le_of_not_le hm)).symm
    /-
      🎉 no goals
    -/
  /-
    F✝ : Type u_1
    inst✝² : Field F✝
    q✝ : Nat
    F : Type u_1
    inst✝¹ : Field F
    q : Nat
    hq : NeZero q
    inst✝ : CharP F q
    g g' : Polynomial F
    m m' : Nat
    h_expand : Eq ((Polynomial.expand F (HPow.hPow q m)) g) ((Polynomial.expand F  …
    hg : g.Separable
    hg' : g'.Separable
    hm : LE.le m m'
    ⊢ Eq g.natDegree g'.natDegree
  -/
  obtain ⟨s, rfl⟩ := exists_add_of_le hm
  /-
    case intro
    F✝ : Type u_1
    inst✝² : Field F✝
    q✝ : Nat
    F : Type u_1
    inst✝¹ : Field F
    q : Nat
    hq : NeZero q
    inst✝ : CharP F q
    g g' : Polynomial F
    m : Nat
    hg : g.Separable
    hg' : g'.Separable
    s : Nat
    h_expand : Eq ((Polynomial.expand F (HPow.hPow q m)) g) ((Polynomial.expand F  …
    hm : LE.le m (HAdd.hAdd m s)
    ⊢ Eq g.natDegree g'.natDegree
  -/
  rw [pow_add, expand_mul, expand_inj (pow_pos (NeZero.pos q) m)] at h_expand
  /-
    case intro
    F✝ : Type u_1
    inst✝² : Field F✝
    q✝ : Nat
    F : Type u_1
    inst✝¹ : Field F
    q : Nat
    hq : NeZero q
    inst✝ : CharP F q
    g g' : Polynomial F
    m : Nat
    hg : g.Separable
    hg' : g'.Separable
    s : Nat
    h_expand : Eq g ((Polynomial.expand F (HPow.hPow q s)) g')
    hm : LE.le m (HAdd.hAdd m s)
    ⊢ Eq g.natDegree g'.natDegree
  -/
  subst h_expand
  /-
    case intro
    F✝ : Type u_1
    inst✝² : Field F✝
    q✝ : Nat
    F : Type u_1
    inst✝¹ : Field F
    q : Nat
    hq : NeZero q
    inst✝ : CharP F q
    g' : Polynomial F
    m : Nat
    hg' : g'.Separable
    s : Nat
    hm : LE.le m (HAdd.hAdd m s)
    hg : ((Polynomial.expand F (HPow.hPow q s)) g').Separable
    ⊢ Eq ((Polynomial.expand F (HPow.hPow q s)) g').natDegree g'.natDegree
  -/
  rcases isUnit_or_eq_zero_of_separable_expand q s (NeZero.pos q) hg with (h | rfl)
    /-
      case intro.inl
      F✝ : Type u_1
      inst✝² : Field F✝
      q✝ : Nat
      F : Type u_1
      inst✝¹ : Field F
      q : Nat
      hq : NeZero q
      inst✝ : CharP F q
      g' : Polynomial F
      m : Nat
      hg' : g'.Separable
      s : Nat
      hm : LE.le m (HAdd.hAdd m s)
      hg : ((Polynomial.expand F (HPow.hPow q s)) g').Separable
      h : IsUnit g'
      ⊢ Eq ((Polynomial.expand F (HPow.hPow q s)) g').natDegree g'.natDegree
    -/
  · rw [natDegree_expand, natDegree_eq_zero_of_isUnit h, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      F✝ : Type u_1
      inst✝² : Field F✝
      q✝ : Nat
      F : Type u_1
      inst✝¹ : Field F
      q : Nat
      hq : NeZero q
      inst✝ : CharP F q
      g' : Polynomial F
      m : Nat
      hg' : g'.Separable
      hm : LE.le m (HAdd.hAdd m 0)
      hg : ((Polynomial.expand F (HPow.hPow q 0)) g').Separable
      ⊢ Eq ((Polynomial.expand F (HPow.hPow q 0)) g').natDegree g'.natDegree
    -/
  · rw [natDegree_expand, pow_zero, mul_one]
    /-
      🎉 no goals
    -/


/-- The separable degree equals the degree of any separable contraction, i.e., it is unique. -/
theorem IsSeparableContraction.degree_eq [hF : ExpChar F q] (g : F[X])
    (hg : IsSeparableContraction q f g) : g.natDegree = hf.degree := by
  /-
    F : Type u_1
    inst✝ : Field F
    q : Nat
    f : Polynomial F
    hf : Polynomial.HasSeparableContraction q f
    hF : ExpChar F q
    g : Polynomial F
    hg : Polynomial.IsSeparableContraction q f g
    ⊢ Eq g.natDegree hf.degree
  -/
  cases hF
    /-
      case zero
      F : Type u_1
      inst✝¹ : Field F
      f g : Polynomial F
      inst✝ : CharZero F
      hf : Polynomial.HasSeparableContraction 1 f
      hg : Polynomial.IsSeparableContraction 1 f g
      ⊢ Eq g.natDegree hf.degree
    -/
  · rcases hg with ⟨_, m, hm⟩
    /-
      case zero.intro.intro
      F : Type u_1
      inst✝¹ : Field F
      f g : Polynomial F
      inst✝ : CharZero F
      hf : Polynomial.HasSeparableContraction 1 f
      left✝ : g.Separable
      m : Nat
      hm : Eq ((Polynomial.expand F (HPow.hPow 1 m)) g) f
      ⊢ Eq g.natDegree hf.degree
    -/
    rw [one_pow, expand_one] at hm
    /-
      case zero.intro.intro
      F : Type u_1
      inst✝¹ : Field F
      f g : Polynomial F
      inst✝ : CharZero F
      hf : Polynomial.HasSeparableContraction 1 f
      left✝ : g.Separable
      m : Nat
      hm : Eq g f
      ⊢ Eq g.natDegree hf.degree
    -/
    rw [hf.eq_degree, hm]
    /-
      🎉 no goals
    -/
    /-
      case prime
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      hf : Polynomial.HasSeparableContraction q f
      g : Polynomial F
      hg : Polynomial.IsSeparableContraction q f g
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      ⊢ Eq g.natDegree hf.degree
    -/
  · rcases hg with ⟨hg, m, hm⟩
    /-
      case prime.intro.intro
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      hf : Polynomial.HasSeparableContraction q f
      g : Polynomial F
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      hg : g.Separable
      m : Nat
      hm : Eq ((Polynomial.expand F (HPow.hPow q m)) g) f
      ⊢ Eq g.natDegree hf.degree
    -/
    let g' := Classical.choose hf
    /-
      case prime.intro.intro
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      hf : Polynomial.HasSeparableContraction q f
      g : Polynomial F
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      hg : g.Separable
      m : Nat
      hm : Eq ((Polynomial.expand F (HPow.hPow q m)) g) f
      g' : Polynomial F := Classical.choose hf
      ⊢ Eq g.natDegree hf.degree
    -/
    obtain ⟨hg', m', hm'⟩ := Classical.choose_spec hf
    /-
      case prime.intro.intro.intro.intro
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      hf : Polynomial.HasSeparableContraction q f
      g : Polynomial F
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      hg : g.Separable
      m : Nat
      hm : Eq ((Polynomial.expand F (HPow.hPow q m)) g) f
      g' : Polynomial F := Classical.choose hf
      hg' : (Classical.choose hf).Separable
      m' : Nat
      hm' : Eq ((Polynomial.expand F (HPow.hPow q m')) (Classical.choose hf)) f
      ⊢ Eq g.natDegree hf.degree
    -/
    haveI : Fact q.Prime := ⟨by assumption⟩
    /-
      case prime.intro.intro.intro.intro
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      hf : Polynomial.HasSeparableContraction q f
      g : Polynomial F
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      hg : g.Separable
      m : Nat
      hm : Eq ((Polynomial.expand F (HPow.hPow q m)) g) f
      g' : Polynomial F := Classical.choose hf
      hg' : (Classical.choose hf).Separable
      m' : Nat
      hm' : Eq ((Polynomial.expand F (HPow.hPow q m')) (Classical.choose hf)) f
      this : Fact (Nat.Prime q)
      ⊢ Eq g.natDegree hf.degree
    -/
    refine contraction_degree_eq_or_insep q g g' m m' ?_ hg hg'
    /-
      case prime.intro.intro.intro.intro
      F : Type u_1
      inst✝ : Field F
      q : Nat
      f : Polynomial F
      hf : Polynomial.HasSeparableContraction q f
      g : Polynomial F
      hprime✝ : Nat.Prime q
      hchar✝ : CharP F q
      hg : g.Separable
      m : Nat
      hm : Eq ((Polynomial.expand F (HPow.hPow q m)) g) f
      g' : Polynomial F := Classical.choose hf
      hg' : (Classical.choose hf).Separable
      m' : Nat
      hm' : Eq ((Polynomial.expand F (HPow.hPow q m')) (Classical.choose hf)) f
      this : Fact (Nat.Prime q)
      ⊢ Eq ((Polynomial.expand F (HPow.hPow q m)) g) ((Polynomial.expand F (HPow.hPo …
    -/
    rw [hm, hm']
    /-
      🎉 no goals
    -/


