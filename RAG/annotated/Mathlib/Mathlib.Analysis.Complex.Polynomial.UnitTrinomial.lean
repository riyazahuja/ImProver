/-- A unit trinomial is irreducible if it has no complex roots in common with its mirror. -/
theorem irreducible_of_coprime' (hp : IsUnitTrinomial p)
    (h : ∀ z : ℂ, ¬(aeval z p = 0 ∧ aeval z (mirror p) = 0)) : Irreducible p := by
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    ⊢ Irreducible p
  -/
  refine hp.irreducible_of_coprime fun q hq hq' => ?_
  suffices ¬0 < q.natDegree by
    rcases hq with ⟨p, rfl⟩
    replace hp := hp.leadingCoeff_isUnit
    rw [leadingCoeff_mul] at hp
    replace hp := isUnit_of_mul_isUnit_left hp
    rw [not_lt, Nat.le_zero] at this
    rwa [eq_C_of_natDegree_eq_zero this, isUnit_C, ← this]
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    q : Polynomial Int
    hq : Dvd.dvd q p
    hq' : Dvd.dvd q p.mirror
    ⊢ Not (LT.lt 0 q.natDegree)
  -/
  intro hq''
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    q : Polynomial Int
    hq : Dvd.dvd q p
    hq' : Dvd.dvd q p.mirror
    hq'' : LT.lt 0 q.natDegree
    ⊢ False
  -/
  rw [natDegree_pos_iff_degree_pos] at hq''
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    q : Polynomial Int
    hq : Dvd.dvd q p
    hq' : Dvd.dvd q p.mirror
    hq'' : LT.lt 0 q.degree
    ⊢ False
  -/
  rw [← degree_map_eq_of_injective (algebraMap ℤ ℂ).injective_int] at hq''
  /-
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    q : Polynomial Int
    hq : Dvd.dvd q p
    hq' : Dvd.dvd q p.mirror
    hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
    ⊢ False
  -/
  cases' Complex.exists_root hq'' with z hz
  /-
    case intro
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    q : Polynomial Int
    hq : Dvd.dvd q p
    hq' : Dvd.dvd q p.mirror
    hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
    z : Complex
    hz : (Polynomial.map (algebraMap Int Complex) q).IsRoot z
    ⊢ False
  -/
  rw [IsRoot, eval_map, ← aeval_def] at hz
  /-
    case intro
    p : Polynomial Int
    hp : p.IsUnitTrinomial
    h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
    q : Polynomial Int
    hq : Dvd.dvd q p
    hq' : Dvd.dvd q p.mirror
    hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
    z : Complex
    hz : Eq ((Polynomial.aeval z) q) 0
    ⊢ False
  -/
  refine h z ⟨?_, ?_⟩
    /-
      case intro.refine_1
      p : Polynomial Int
      hp : p.IsUnitTrinomial
      h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
      q : Polynomial Int
      hq : Dvd.dvd q p
      hq' : Dvd.dvd q p.mirror
      hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
      z : Complex
      hz : Eq ((Polynomial.aeval z) q) 0
      ⊢ Eq ((Polynomial.aeval z) p) 0
    -/
  · cases' hq with g' hg'
    /-
      case intro.refine_1.intro
      p : Polynomial Int
      hp : p.IsUnitTrinomial
      h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
      q : Polynomial Int
      hq' : Dvd.dvd q p.mirror
      hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
      z : Complex
      hz : Eq ((Polynomial.aeval z) q) 0
      g' : Polynomial Int
      hg' : Eq p (HMul.hMul q g')
      ⊢ Eq ((Polynomial.aeval z) p) 0
    -/
    rw [hg', aeval_mul, hz, zero_mul]
    /-
      🎉 no goals
    -/
    /-
      case intro.refine_2
      p : Polynomial Int
      hp : p.IsUnitTrinomial
      h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
      q : Polynomial Int
      hq : Dvd.dvd q p
      hq' : Dvd.dvd q p.mirror
      hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
      z : Complex
      hz : Eq ((Polynomial.aeval z) q) 0
      ⊢ Eq ((Polynomial.aeval z) p.mirror) 0
    -/
  · cases' hq' with g' hg'
    /-
      case intro.refine_2.intro
      p : Polynomial Int
      hp : p.IsUnitTrinomial
      h : ∀ (z : Complex), Not (And (Eq ((Polynomial.aeval z) p) 0) (Eq ((Polynomial …
      q : Polynomial Int
      hq : Dvd.dvd q p
      hq'' : LT.lt 0 (Polynomial.map (algebraMap Int Complex) q).degree
      z : Complex
      hz : Eq ((Polynomial.aeval z) q) 0
      g' : Polynomial Int
      hg' : Eq p.mirror (HMul.hMul q g')
      ⊢ Eq ((Polynomial.aeval z) p.mirror) 0
    -/
    rw [hg', aeval_mul, hz, zero_mul]
    /-
      🎉 no goals
    -/


