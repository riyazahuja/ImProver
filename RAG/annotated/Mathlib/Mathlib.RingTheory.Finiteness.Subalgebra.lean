theorem fg_bot_toSubmodule  : (⊥ : Subalgebra R A).toSubmodule.FG :=
           /-
             R : Type u_1
             A : Type u_2
             inst✝² : CommSemiring R
             inst✝¹ : Semiring A
             inst✝ : Algebra R A
             ⊢ Eq (Submodule.span R ↑(Singleton.singleton 1)) (Subalgebra.toSubmodule Bot.b …
           -/
  ⟨{1}, by simp [Algebra.toSubmodule_bot, one_eq_span]⟩
           /-
             🎉 no goals
           -/


instance finite_bot : Module.Finite R (⊥ : Subalgebra R A) :=
  Module.Finite.range (Algebra.linearMap R A)


theorem fg_unit {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A] (I : (Submodule R A)ˣ) :
    (I : Submodule R A).FG := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    I : Units (Submodule R A)
    ⊢ (↑I).FG
  -/
  obtain ⟨T, T', hT, hT', one_mem⟩ := mem_span_mul_finite_of_mem_mul (I.mul_inv ▸ one_le.mp le_rfl)
  /-
    case intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    I : Units (Submodule R A)
    T T' : Finset A
    hT : HasSubset.Subset ↑T ↑↑I
    hT' : HasSubset.Subset ↑T' ↑↑(Inv.inv I)
    one_mem : Membership.mem (Submodule.span R (HMul.hMul ↑T ↑T')) 1
    ⊢ (↑I).FG
  -/
  refine ⟨T, span_eq_of_le _ hT ?_⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    I : Units (Submodule R A)
    T T' : Finset A
    hT : HasSubset.Subset ↑T ↑↑I
    hT' : HasSubset.Subset ↑T' ↑↑(Inv.inv I)
    one_mem : Membership.mem (Submodule.span R (HMul.hMul ↑T ↑T')) 1
    ⊢ LE.le (↑I) (Submodule.span R ↑T)
  -/
  rw [← one_mul I, ← mul_one (span R (T : Set A))]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    I : Units (Submodule R A)
    T T' : Finset A
    hT : HasSubset.Subset ↑T ↑↑I
    hT' : HasSubset.Subset ↑T' ↑↑(Inv.inv I)
    one_mem : Membership.mem (Submodule.span R (HMul.hMul ↑T ↑T')) 1
    ⊢ LE.le (↑(HMul.hMul 1 I)) (HMul.hMul (Submodule.span R ↑T) 1)
  -/
  conv_rhs => rw [← I.inv_mul, ← mul_assoc]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    I : Units (Submodule R A)
    T T' : Finset A
    hT : HasSubset.Subset ↑T ↑↑I
    hT' : HasSubset.Subset ↑T' ↑↑(Inv.inv I)
    one_mem : Membership.mem (Submodule.span R (HMul.hMul ↑T ↑T')) 1
    ⊢ LE.le (↑(HMul.hMul 1 I)) (HMul.hMul (HMul.hMul (Submodule.span R ↑T) ↑(Inv.i …
  -/
  refine mul_le_mul_left (le_trans ?_ <| mul_le_mul_right <| span_le.mpr hT')
  /-
    case intro.intro.intro.intro
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    I : Units (Submodule R A)
    T T' : Finset A
    hT : HasSubset.Subset ↑T ↑↑I
    hT' : HasSubset.Subset ↑T' ↑↑(Inv.inv I)
    one_mem : Membership.mem (Submodule.span R (HMul.hMul ↑T ↑T')) 1
    ⊢ LE.le (↑1) (HMul.hMul (Submodule.span R ↑T) (Submodule.span R ↑T'))
  -/
  rwa [Units.val_one, span_mul_span, one_le]
  /-
    🎉 no goals
  -/


theorem fg_of_isUnit {R A : Type*} [CommSemiring R] [Semiring A] [Algebra R A] {I : Submodule R A}
    (hI : IsUnit I) : I.FG :=
  fg_unit hI.unit


theorem FG.mul (hm : M.FG) (hn : N.FG) : (M * N).FG := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    M N : Submodule R A
    hm : M.FG
    hn : N.FG
    ⊢ (HMul.hMul M N).FG
  -/
  rw [mul_eq_map₂]; exact hm.map₂ _ hn
                    /-
                      🎉 no goals
                    -/


theorem FG.pow (h : M.FG) (n : ℕ) : (M ^ n).FG :=
                       /-
                         R : Type u_1
                         A : Type u_2
                         inst✝² : CommSemiring R
                         inst✝¹ : Semiring A
                         inst✝ : Algebra R A
                         M : Submodule R A
                         h : M.FG
                         n : Nat
                         ⊢ Eq (Submodule.span R ↑(Singleton.singleton 1)) (HPow.hPow M Nat.zero)
                       -/
                       /-
                         🎉 no goals
                       -/
  Nat.recOn n ⟨{1}, by simp [one_eq_span]⟩ fun n ih => by simpa [pow_succ] using ih.mul h
                                                          /-
                                                            🎉 no goals
                                                          -/


