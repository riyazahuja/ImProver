theorem eigenspace_aeval_polynomial_degree_1 (f : End K V) (q : K[X]) (hq : degree q = 1) :
    eigenspace f (-q.coeff 0 / q.leadingCoeff) = LinearMap.ker (aeval f q) :=
  calc
    eigenspace f (-q.coeff 0 / q.leadingCoeff)
    _ = LinearMap.ker (q.leadingCoeff • f - algebraMap K (End K V) (-q.coeff 0)) := by
          /-
            K : Type v
            V : Type w
            inst✝² : Field K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            f : Module.End K V
            q : Polynomial K
            hq : Eq q.degree 1
            ⊢ Eq (f.eigenspace (HDiv.hDiv (Neg.neg (q.coeff 0)) q.leadingCoeff)) (LinearMa …
          -/
          rw [eigenspace_div]
          /-
            case hb
            K : Type v
            V : Type w
            inst✝² : Field K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            f : Module.End K V
            q : Polynomial K
            hq : Eq q.degree 1
            ⊢ Ne q.leadingCoeff 0
          -/
          intro h
          /-
            case hb
            K : Type v
            V : Type w
            inst✝² : Field K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            f : Module.End K V
            q : Polynomial K
            hq : Eq q.degree 1
            h : Eq q.leadingCoeff 0
            ⊢ False
          -/
          rw [leadingCoeff_eq_zero_iff_deg_eq_bot.1 h] at hq
          /-
            case hb
            K : Type v
            V : Type w
            inst✝² : Field K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            f : Module.End K V
            q : Polynomial K
            hq : Eq Bot.bot 1
            h : Eq q.leadingCoeff 0
            ⊢ False
          -/
          cases hq
          /-
            🎉 no goals
          -/
    _ = LinearMap.ker (aeval f (C q.leadingCoeff * X + C (q.coeff 0))) := by
          /-
            K : Type v
            V : Type w
            inst✝² : Field K
            inst✝¹ : AddCommGroup V
            inst✝ : Module K V
            f : Module.End K V
            q : Polynomial K
            hq : Eq q.degree 1
            ⊢ Eq (LinearMap.ker (HSub.hSub (HSMul.hSMul q.leadingCoeff f) ((algebraMap K ( …
          -/
          rw [C_mul', aeval_def]; simp [algebraMap, Algebra.toRingHom]
                                  /-
                                    🎉 no goals
                                  -/
                                        /-
                                          K : Type v
                                          V : Type w
                                          inst✝² : Field K
                                          inst✝¹ : AddCommGroup V
                                          inst✝ : Module K V
                                          f : Module.End K V
                                          q : Polynomial K
                                          hq : Eq q.degree 1
                                          ⊢ Eq (LinearMap.ker ((Polynomial.aeval f) (HAdd.hAdd (HMul.hMul (Polynomial.C  …
                                        -/
    _ = LinearMap.ker (aeval f q) := by rwa [← eq_X_add_C_of_degree_eq_one]
                                        /-
                                          🎉 no goals
                                        -/


theorem ker_aeval_ring_hom'_unit_polynomial (f : End K V) (c : K[X]ˣ) :
    LinearMap.ker (aeval f (c : K[X])) = ⊥ := by
  /-
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    c : Units (Polynomial K)
    ⊢ Eq (LinearMap.ker ((Polynomial.aeval f) ↑c)) Bot.bot
  -/
  rw [Polynomial.eq_C_of_degree_eq_zero (degree_coe_units c)]
  /-
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    c : Units (Polynomial K)
    ⊢ Eq (LinearMap.ker ((Polynomial.aeval f) (Polynomial.C ((↑c).coeff 0)))) Bot. …
  -/
  simp only [aeval_def, eval₂_C]
  /-
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    c : Units (Polynomial K)
    ⊢ Eq (LinearMap.ker ((algebraMap K (Module.End K V)) ((↑c).coeff 0))) Bot.bot
  -/
  apply ker_algebraMap_end
  /-
    case ha
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    c : Units (Polynomial K)
    ⊢ Ne ((↑c).coeff 0) 0
  -/
  apply coeff_coe_units_zero_ne_zero c
  /-
    🎉 no goals
  -/


theorem aeval_apply_of_hasEigenvector {f : End K V} {p : K[X]} {μ : K} {x : V}
    (h : f.HasEigenvector μ x) : aeval f p x = p.eval μ • x := by
  /-
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    p : Polynomial K
    μ : K
    x : V
    h : f.HasEigenvector μ x
    ⊢ Eq (((Polynomial.aeval f) p) x) (HSMul.hSMul (Polynomial.eval μ p) x)
  -/
  refine p.induction_on ?_ ?_ ?_
    /-
      case refine_1
      K : Type v
      V : Type w
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      p : Polynomial K
      μ : K
      x : V
      h : f.HasEigenvector μ x
      ⊢ ∀ (a : K), Eq (((Polynomial.aeval f) (Polynomial.C a)) x) (HSMul.hSMul (Poly …
    -/
  · intro a; simp [Module.algebraMap_end_apply]
             /-
               🎉 no goals
             -/
    /-
      case refine_2
      K : Type v
      V : Type w
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      p : Polynomial K
      μ : K
      x : V
      h : f.HasEigenvector μ x
      ⊢ ∀ (p q : Polynomial K), Eq (((Polynomial.aeval f) p) x) (HSMul.hSMul (Polyno …
    -/
  · intro p q hp hq; simp [hp, hq, add_smul]
                     /-
                       🎉 no goals
                     -/
    /-
      case refine_3
      K : Type v
      V : Type w
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      p : Polynomial K
      μ : K
      x : V
      h : f.HasEigenvector μ x
      ⊢ ∀ (n : Nat) (a : K), Eq (((Polynomial.aeval f) (HMul.hMul (Polynomial.C a) ( …
    -/
  · intro n a hna
    /-
      case refine_3
      K : Type v
      V : Type w
      inst✝² : Field K
      inst✝¹ : AddCommGroup V
      inst✝ : Module K V
      f : Module.End K V
      p : Polynomial K
      μ : K
      x : V
      h : f.HasEigenvector μ x
      n : Nat
      a : K
      hna : Eq (((Polynomial.aeval f) (HMul.hMul (Polynomial.C a) (HPow.hPow Polynom …
      ⊢ Eq (((Polynomial.aeval f) (HMul.hMul (Polynomial.C a) (HPow.hPow Polynomial. …
    -/
    rw [mul_comm, pow_succ', mul_assoc, map_mul, LinearMap.mul_apply, mul_comm, hna]
    simp only [mem_eigenspace_iff.1 h.1, smul_smul, aeval_X, eval_mul, eval_C, eval_pow, eval_X,
      LinearMap.map_smulₛₗ, RingHom.id_apply, mul_comm]


theorem isRoot_of_hasEigenvalue {f : End K V} {μ : K} (h : f.HasEigenvalue μ) :
    (minpoly K f).IsRoot μ := by
  /-
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    μ : K
    h : f.HasEigenvalue μ
    ⊢ (minpoly K f).IsRoot μ
  -/
  rcases (Submodule.ne_bot_iff _).1 h with ⟨w, ⟨H, ne0⟩⟩
  /-
    case intro.intro
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    μ : K
    h : f.HasEigenvalue μ
    w : V
    H : Membership.mem ((f.genEigenspace μ) 1) w
    ne0 : Ne w 0
    ⊢ (minpoly K f).IsRoot μ
  -/
  refine Or.resolve_right (smul_eq_zero.1 ?_) ne0
  /-
    case intro.intro
    K : Type v
    V : Type w
    inst✝² : Field K
    inst✝¹ : AddCommGroup V
    inst✝ : Module K V
    f : Module.End K V
    μ : K
    h : f.HasEigenvalue μ
    w : V
    H : Membership.mem ((f.genEigenspace μ) 1) w
    ne0 : Ne w 0
    ⊢ Eq (HSMul.hSMul (Polynomial.eval μ (minpoly K f)) w) 0
  -/
  simp [← aeval_apply_of_hasEigenvector ⟨H, ne0⟩, minpoly.aeval K f]
  /-
    🎉 no goals
  -/


theorem hasEigenvalue_of_isRoot (h : (minpoly K f).IsRoot μ) : f.HasEigenvalue μ := by
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    ⊢ f.HasEigenvalue μ
  -/
  cases' dvd_iff_isRoot.2 h with p hp
  /-
    case intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    ⊢ f.HasEigenvalue μ
  -/
  rw [hasEigenvalue_iff, eigenspace_def]
  /-
    case intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    ⊢ Ne (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot
  -/
  intro con
  /-
    case intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    con : Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot
    ⊢ False
  -/
  cases' (LinearMap.isUnit_iff_ker_eq_bot _).2 con with u hu
  have p_ne_0 : p ≠ 0 := by
    intro con
    apply minpoly.ne_zero (Algebra.IsIntegral.isIntegral (R := K) f)
    rw [hp, con, mul_zero]
  have : (aeval f) p = 0 := by
    have h_aeval := minpoly.aeval K f
    revert h_aeval
    simp [hp, ← hu, Algebra.algebraMap_eq_smul_one]
  /-
    case intro.intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    con : Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot
    u : Units (LinearMap (RingHom.id K) V V)
    hu : Eq (↑u) (HSub.hSub f (HSMul.hSMul μ 1))
    p_ne_0 : Ne p 0
    this : Eq ((Polynomial.aeval f) p) 0
    ⊢ False
  -/
  have h_deg := minpoly.degree_le_of_ne_zero K f p_ne_0 this
  /-
    case intro.intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    con : Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot
    u : Units (LinearMap (RingHom.id K) V V)
    hu : Eq (↑u) (HSub.hSub f (HSMul.hSMul μ 1))
    p_ne_0 : Ne p 0
    this : Eq ((Polynomial.aeval f) p) 0
    h_deg : LE.le (minpoly K f).degree p.degree
    ⊢ False
  -/
  rw [hp, degree_mul, degree_X_sub_C, Polynomial.degree_eq_natDegree p_ne_0] at h_deg
  /-
    case intro.intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    con : Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot
    u : Units (LinearMap (RingHom.id K) V V)
    hu : Eq (↑u) (HSub.hSub f (HSMul.hSMul μ 1))
    p_ne_0 : Ne p 0
    this : Eq ((Polynomial.aeval f) p) 0
    h_deg : LE.le (HAdd.hAdd 1 ↑p.natDegree) ↑p.natDegree
    ⊢ False
  -/
  norm_cast at h_deg
  /-
    case intro.intro
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    μ : K
    h : (minpoly K f).IsRoot μ
    p : Polynomial K
    hp : Eq (minpoly K f) (HMul.hMul (HSub.hSub Polynomial.X (Polynomial.C μ)) p)
    con : Eq (LinearMap.ker (HSub.hSub f (HSMul.hSMul μ 1))) Bot.bot
    u : Units (LinearMap (RingHom.id K) V V)
    hu : Eq (↑u) (HSub.hSub f (HSMul.hSMul μ 1))
    p_ne_0 : Ne p 0
    this : Eq ((Polynomial.aeval f) p) 0
    h_deg : LE.le (HAdd.hAdd 1 p.natDegree) p.natDegree
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


theorem hasEigenvalue_iff_isRoot : f.HasEigenvalue μ ↔ (minpoly K f).IsRoot μ :=
  ⟨isRoot_of_hasEigenvalue, hasEigenvalue_of_isRoot⟩


lemma finite_hasEigenvalue : Set.Finite f.HasEigenvalue := by
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    ⊢ Set.Finite f.HasEigenvalue
  -/
  have h : minpoly K f ≠ 0 := minpoly.ne_zero (Algebra.IsIntegral.isIntegral (R := K) f)
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    h : Ne (minpoly K f) 0
    ⊢ Set.Finite f.HasEigenvalue
  -/
  convert (minpoly K f).rootSet_finite K
  /-
    case h.e'_2
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    h : Ne (minpoly K f) 0
    ⊢ Eq f.HasEigenvalue ((minpoly K f).rootSet K)
  -/
  ext μ
  /-
    case h.e'_2.h
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    h : Ne (minpoly K f) 0
    μ : K
    ⊢ Iff (Membership.mem f.HasEigenvalue μ) (Membership.mem ((minpoly K f).rootSe …
  -/
  change f.HasEigenvalue μ ↔ _
  /-
    case h.e'_2.h
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    h : Ne (minpoly K f) 0
    μ : K
    ⊢ Iff (f.HasEigenvalue μ) (Membership.mem ((minpoly K f).rootSet K) μ)
  -/
  rw [hasEigenvalue_iff_isRoot, mem_rootSet_of_ne h, IsRoot, coe_aeval_eq_eval]
  /-
    🎉 no goals
  -/


/-- An endomorphism of a finite-dimensional vector space has finitely many eigenvalues. -/
noncomputable instance : Fintype f.Eigenvalues :=
  Set.Finite.fintype f.finite_hasEigenvalue


/-- An endomorphism of a finite-dimensional vector space has a finite spectrum. -/
theorem Module.End.finite_spectrum {K : Type v} {V : Type w} [Field K] [AddCommGroup V]
    [Module K V] [FiniteDimensional K V] (f : Module.End K V) :
    Set.Finite (spectrum K f) := by
  /-
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    ⊢ (spectrum K f).Finite
  -/
  convert f.finite_hasEigenvalue
  /-
    case h.e'_2.h.e
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f : Module.End K V
    ⊢ Eq (spectrum K) Module.End.HasEigenvalue
  -/
  ext f x
  /-
    case h.e'_2.h.e.h.h
    K : Type v
    V : Type w
    inst✝³ : Field K
    inst✝² : AddCommGroup V
    inst✝¹ : Module K V
    inst✝ : FiniteDimensional K V
    f✝ f : Module.End K V
    x : K
    ⊢ Iff (Membership.mem (spectrum K f) x) (Membership.mem f.HasEigenvalue x)
  -/
  exact Module.End.hasEigenvalue_iff_mem_spectrum.symm
  /-
    🎉 no goals
  -/


/-- An n x n matrix over a field has a finite spectrum. -/
theorem Matrix.finite_spectrum (A : Matrix n n R) : Set.Finite (spectrum R A) := by
  /-
    n : Type u_1
    R : Type u_2
    inst✝² : Field R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n R
    ⊢ (spectrum R A).Finite
  -/
  rw [← AlgEquiv.spectrum_eq (Matrix.toLinAlgEquiv <| Pi.basisFun R n) A]
  /-
    n : Type u_1
    R : Type u_2
    inst✝² : Field R
    inst✝¹ : Fintype n
    inst✝ : DecidableEq n
    A : Matrix n n R
    ⊢ (spectrum R ((Matrix.toLinAlgEquiv (Pi.basisFun R n)) A)).Finite
  -/
  exact Module.End.finite_spectrum _
  /-
    🎉 no goals
  -/


instance Matrix.instFiniteSpectrum (A : Matrix n n R) : Finite (spectrum R A) :=
  Set.finite_coe_iff.mpr (Matrix.finite_spectrum A)


