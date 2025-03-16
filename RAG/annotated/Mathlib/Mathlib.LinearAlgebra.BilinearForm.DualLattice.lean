/-- The dual submodule of a submodule with respect to a bilinear form. -/
def dualSubmodule (N : Submodule R M) : Submodule R M where
  carrier := { x | ∀ y ∈ N, B x y ∈ (1 : Submodule R S) }
                                  /-
                                    R : Type ?u.3514
                                    S : Type ?u.3517
                                    M : Type ?u.3520
                                    inst✝⁶ : CommRing R
                                    inst✝⁵ : Field S
                                    inst✝⁴ : AddCommGroup M
                                    inst✝³ : Algebra R S
                                    inst✝² : Module R M
                                    inst✝¹ : Module S M
                                    inst✝ : IsScalarTower R S M
                                    B : LinearMap.BilinForm S M
                                    N : Submodule R M
                                    a b : M
                                    ha : Membership.mem (setOf fun x => ∀ (y : M), Membership.mem N y → Membership …
                                    hb : Membership.mem (setOf fun x => ∀ (y : M), Membership.mem N y → Membership …
                                    y : M
                                    hy : Membership.mem N y
                                    ⊢ Membership.mem 1 ((B (HAdd.hAdd a b)) y)
                                  -/
  add_mem' {a b} ha hb y hy := by simpa using add_mem (ha y hy) (hb y hy)
                                  /-
                                    🎉 no goals
                                  -/
                      /-
                        R : Type ?u.3514
                        S : Type ?u.3517
                        M : Type ?u.3520
                        inst✝⁶ : CommRing R
                        inst✝⁵ : Field S
                        inst✝⁴ : AddCommGroup M
                        inst✝³ : Algebra R S
                        inst✝² : Module R M
                        inst✝¹ : Module S M
                        inst✝ : IsScalarTower R S M
                        B : LinearMap.BilinForm S M
                        N : Submodule R M
                        y : M
                        x✝ : Membership.mem N y
                        ⊢ Membership.mem 1 ((B 0) y)
                      -/
  zero_mem' y _ := by rw [B.zero_left]; exact zero_mem _
                                        /-
                                          🎉 no goals
                                        -/
  smul_mem' r a ha y hy := by
    /-
      R : Type ?u.3514
      S : Type ?u.3517
      M : Type ?u.3520
      inst✝⁶ : CommRing R
      inst✝⁵ : Field S
      inst✝⁴ : AddCommGroup M
      inst✝³ : Algebra R S
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      N : Submodule R M
      r : R
      a : M
      ha : Membership.mem { carrier := setOf fun x => ∀ (y : M), Membership.mem N y  …
      y : M
      hy : Membership.mem N y
      ⊢ Membership.mem 1 ((B (HSMul.hSMul r a)) y)
    -/
    convert (1 : Submodule R S).smul_mem r (ha y hy)
    /-
      case h.e'_5
      R : Type ?u.3514
      S : Type ?u.3517
      M : Type ?u.3520
      inst✝⁶ : CommRing R
      inst✝⁵ : Field S
      inst✝⁴ : AddCommGroup M
      inst✝³ : Algebra R S
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      N : Submodule R M
      r : R
      a : M
      ha : Membership.mem { carrier := setOf fun x => ∀ (y : M), Membership.mem N y  …
      y : M
      hy : Membership.mem N y
      ⊢ Eq ((B (HSMul.hSMul r a)) y) (HSMul.hSMul r ((B a) y))
    -/
    rw [← IsScalarTower.algebraMap_smul S r a]
    /-
      case h.e'_5
      R : Type ?u.3514
      S : Type ?u.3517
      M : Type ?u.3520
      inst✝⁶ : CommRing R
      inst✝⁵ : Field S
      inst✝⁴ : AddCommGroup M
      inst✝³ : Algebra R S
      inst✝² : Module R M
      inst✝¹ : Module S M
      inst✝ : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      N : Submodule R M
      r : R
      a : M
      ha : Membership.mem { carrier := setOf fun x => ∀ (y : M), Membership.mem N y  …
      y : M
      hy : Membership.mem N y
      ⊢ Eq ((B (HSMul.hSMul ((algebraMap R S) r) a)) y) (HSMul.hSMul r ((B a) y))
    -/
    simp only [algebraMap_smul, map_smul_of_tower, LinearMap.smul_apply]
    /-
      🎉 no goals
    -/


lemma mem_dualSubmodule {N : Submodule R M} {x} :
    x ∈ B.dualSubmodule N ↔ ∀ y ∈ N, B x y ∈ (1 : Submodule R S) := Iff.rfl


lemma le_flip_dualSubmodule {N₁ N₂ : Submodule R M} :
    N₁ ≤ B.flip.dualSubmodule N₂ ↔ N₂ ≤ B.dualSubmodule N₁ := by
  /-
    R : Type u_1
    S : Type u_3
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : Field S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Algebra R S
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    N₁ N₂ : Submodule R M
    ⊢ Iff (LE.le N₁ (B.flip.dualSubmodule N₂)) (LE.le N₂ (B.dualSubmodule N₁))
  -/
  show (∀ (x : M), x ∈ N₁ → _) ↔ ∀ (x : M), x ∈ N₂ → _
  /-
    R : Type u_1
    S : Type u_3
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : Field S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Algebra R S
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    N₁ N₂ : Submodule R M
    ⊢ Iff (∀ (x : M), Membership.mem N₁ x → Membership.mem (B.flip.dualSubmodule N …
  -/
  simp only [mem_dualSubmodule, Submodule.mem_one, flip_apply]
  /-
    R : Type u_1
    S : Type u_3
    M : Type u_2
    inst✝⁶ : CommRing R
    inst✝⁵ : Field S
    inst✝⁴ : AddCommGroup M
    inst✝³ : Algebra R S
    inst✝² : Module R M
    inst✝¹ : Module S M
    inst✝ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    N₁ N₂ : Submodule R M
    ⊢ Iff (∀ (x : M), Membership.mem N₁ x → ∀ (y : M), Membership.mem N₂ y → Exist …
  -/
  exact forall₂_swap
  /-
    🎉 no goals
  -/


/-- The natural paring of `B.dualSubmodule N` and `N`.
This is bundled as a bilinear map in `BilinForm.dualSubmoduleToDual`. -/
noncomputable
def dualSubmoduleParing {N : Submodule R M} (x : B.dualSubmodule N) (y : N) : R :=
  (Submodule.mem_one.mp <| x.prop y y.prop).choose


@[simp]
lemma dualSubmoduleParing_spec {N : Submodule R M} (x : B.dualSubmodule N) (y : N) :
    algebraMap R S (B.dualSubmoduleParing x y) = B x y :=
  (Submodule.mem_one.mp <| x.prop y y.prop).choose_spec


/-- The natural paring of `B.dualSubmodule N` and `N`. -/
-- TODO: Show that this is perfect when `N` is a lattice and `B` is nondegenerate.
@[simps]
noncomputable
def dualSubmoduleToDual [NoZeroSMulDivisors R S] (N : Submodule R M) :
    B.dualSubmodule N →ₗ[R] Module.Dual R N :=
  { toFun := fun x ↦
    { toFun := B.dualSubmoduleParing x
                                                                            /-
                                                                              R : Type ?u.31327
                                                                              S : Type ?u.31330
                                                                              M : Type ?u.31333
                                                                              inst✝⁷ : CommRing R
                                                                              inst✝⁶ : Field S
                                                                              inst✝⁵ : AddCommGroup M
                                                                              inst✝⁴ : Algebra R S
                                                                              inst✝³ : Module R M
                                                                              inst✝² : Module S M
                                                                              inst✝¹ : IsScalarTower R S M
                                                                              B : LinearMap.BilinForm S M
                                                                              inst✝ : NoZeroSMulDivisors R S
                                                                              N : Submodule R M
                                                                              x✝ : Subtype fun x => Membership.mem (B.dualSubmodule N) x
                                                                              x y : Subtype fun x => Membership.mem N x
                                                                              ⊢ Eq ((algebraMap R S) (B.dualSubmoduleParing x✝ (HAdd.hAdd x y))) ((algebraMa …
                                                                            -/
      map_add' := fun x y ↦ NoZeroSMulDivisors.algebraMap_injective R S (by simp)
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
      map_smul' := fun r m ↦ NoZeroSMulDivisors.algebraMap_injective R S
            /-
              R : Type ?u.31327
              S : Type ?u.31330
              M : Type ?u.31333
              inst✝⁷ : CommRing R
              inst✝⁶ : Field S
              inst✝⁵ : AddCommGroup M
              inst✝⁴ : Algebra R S
              inst✝³ : Module R M
              inst✝² : Module S M
              inst✝¹ : IsScalarTower R S M
              B : LinearMap.BilinForm S M
              inst✝ : NoZeroSMulDivisors R S
              N : Submodule R M
              x : Subtype fun x => Membership.mem (B.dualSubmodule N) x
              r : R
              m : Subtype fun x => Membership.mem N x
              ⊢ Eq ((algebraMap R S) ({ toFun := B.dualSubmoduleParing x, map_add' := ⋯ }.to …
            -/
        (by simp [← Algebra.smul_def]) }
            /-
              🎉 no goals
            -/
    map_add' := fun x y ↦ LinearMap.ext fun z ↦ NoZeroSMulDivisors.algebraMap_injective R S
          /-
            R : Type ?u.31327
            S : Type ?u.31330
            M : Type ?u.31333
            inst✝⁷ : CommRing R
            inst✝⁶ : Field S
            inst✝⁵ : AddCommGroup M
            inst✝⁴ : Algebra R S
            inst✝³ : Module R M
            inst✝² : Module S M
            inst✝¹ : IsScalarTower R S M
            B : LinearMap.BilinForm S M
            inst✝ : NoZeroSMulDivisors R S
            N : Submodule R M
            x y : Subtype fun x => Membership.mem (B.dualSubmodule N) x
            z : Subtype fun x => Membership.mem N x
            ⊢ Eq ((algebraMap R S) (((fun x => { toFun := B.dualSubmoduleParing x, map_add …
          -/
      (by simp)
          /-
            🎉 no goals
          -/
    map_smul' := fun r x ↦ LinearMap.ext fun y ↦ NoZeroSMulDivisors.algebraMap_injective R S
          /-
            R : Type ?u.31327
            S : Type ?u.31330
            M : Type ?u.31333
            inst✝⁷ : CommRing R
            inst✝⁶ : Field S
            inst✝⁵ : AddCommGroup M
            inst✝⁴ : Algebra R S
            inst✝³ : Module R M
            inst✝² : Module S M
            inst✝¹ : IsScalarTower R S M
            B : LinearMap.BilinForm S M
            inst✝ : NoZeroSMulDivisors R S
            N : Submodule R M
            r : R
            x : Subtype fun x => Membership.mem (B.dualSubmodule N) x
            y : Subtype fun x => Membership.mem N x
            ⊢ Eq ((algebraMap R S) (({ toFun := fun x => { toFun := B.dualSubmoduleParing  …
          -/
      (by simp [← Algebra.smul_def]) }
          /-
            🎉 no goals
          -/


lemma dualSubmoduleToDual_injective (hB : B.Nondegenerate) [NoZeroSMulDivisors R S]
    (N : Submodule R M) (hN : Submodule.span S (N : Set M) = ⊤) :
    Function.Injective (B.dualSubmoduleToDual N) := by
  /-
    R : Type u_3
    S : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Field S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    hB : B.Nondegenerate
    inst✝ : NoZeroSMulDivisors R S
    N : Submodule R M
    hN : Eq (Submodule.span S ↑N) Top.top
    ⊢ Function.Injective ⇑(B.dualSubmoduleToDual N)
  -/
  intro x y e
  /-
    R : Type u_3
    S : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Field S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    hB : B.Nondegenerate
    inst✝ : NoZeroSMulDivisors R S
    N : Submodule R M
    hN : Eq (Submodule.span S ↑N) Top.top
    x y : Subtype fun x => Membership.mem (B.dualSubmodule N) x
    e : Eq ((B.dualSubmoduleToDual N) x) ((B.dualSubmoduleToDual N) y)
    ⊢ Eq x y
  -/
  ext
  /-
    case a
    R : Type u_3
    S : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Field S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    hB : B.Nondegenerate
    inst✝ : NoZeroSMulDivisors R S
    N : Submodule R M
    hN : Eq (Submodule.span S ↑N) Top.top
    x y : Subtype fun x => Membership.mem (B.dualSubmodule N) x
    e : Eq ((B.dualSubmoduleToDual N) x) ((B.dualSubmoduleToDual N) y)
    ⊢ Eq ↑x ↑y
  -/
  apply LinearMap.ker_eq_bot.mp hB.ker_eq_bot
  /-
    case a.a
    R : Type u_3
    S : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Field S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    hB : B.Nondegenerate
    inst✝ : NoZeroSMulDivisors R S
    N : Submodule R M
    hN : Eq (Submodule.span S ↑N) Top.top
    x y : Subtype fun x => Membership.mem (B.dualSubmodule N) x
    e : Eq ((B.dualSubmoduleToDual N) x) ((B.dualSubmoduleToDual N) y)
    ⊢ Eq (B ↑x) (B ↑y)
  -/
  apply LinearMap.ext_on hN
  /-
    case a.a
    R : Type u_3
    S : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Field S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    hB : B.Nondegenerate
    inst✝ : NoZeroSMulDivisors R S
    N : Submodule R M
    hN : Eq (Submodule.span S ↑N) Top.top
    x y : Subtype fun x => Membership.mem (B.dualSubmodule N) x
    e : Eq ((B.dualSubmoduleToDual N) x) ((B.dualSubmoduleToDual N) y)
    ⊢ Set.EqOn ⇑(B ↑x) ⇑(B ↑y) ↑N
  -/
  intro z hz
  /-
    case a.a
    R : Type u_3
    S : Type u_1
    M : Type u_2
    inst✝⁷ : CommRing R
    inst✝⁶ : Field S
    inst✝⁵ : AddCommGroup M
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module S M
    inst✝¹ : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    hB : B.Nondegenerate
    inst✝ : NoZeroSMulDivisors R S
    N : Submodule R M
    hN : Eq (Submodule.span S ↑N) Top.top
    x y : Subtype fun x => Membership.mem (B.dualSubmodule N) x
    e : Eq ((B.dualSubmoduleToDual N) x) ((B.dualSubmoduleToDual N) y)
    z : M
    hz : Membership.mem (↑N) z
    ⊢ Eq ((B ↑x) z) ((B ↑y) z)
  -/
  simpa using congr_arg (algebraMap R S) (LinearMap.congr_fun e ⟨z, hz⟩)
  /-
    🎉 no goals
  -/


lemma dualSubmodule_span_of_basis {ι} [Finite ι] [DecidableEq ι]
    (hB : B.Nondegenerate) (b : Basis ι S M) :
    B.dualSubmodule (Submodule.span R (Set.range b)) =
      Submodule.span R (Set.range <| B.dualBasis hB b) := by
  /-
    R : Type u_4
    S : Type u_2
    M : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : Field S
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    ι : Type u_1
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    hB : B.Nondegenerate
    b : Basis ι S M
    ⊢ Eq (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) (Submodule.span R (Se …
  -/
  cases nonempty_fintype ι
  /-
    case intro
    R : Type u_4
    S : Type u_2
    M : Type u_3
    inst✝⁸ : CommRing R
    inst✝⁷ : Field S
    inst✝⁶ : AddCommGroup M
    inst✝⁵ : Algebra R S
    inst✝⁴ : Module R M
    inst✝³ : Module S M
    inst✝² : IsScalarTower R S M
    B : LinearMap.BilinForm S M
    ι : Type u_1
    inst✝¹ : Finite ι
    inst✝ : DecidableEq ι
    hB : B.Nondegenerate
    b : Basis ι S M
    val✝ : Fintype ι
    ⊢ Eq (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) (Submodule.span R (Se …
  -/
  apply le_antisymm
    /-
      case intro.a
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      ⊢ LE.le (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) (Submodule.span R  …
    -/
  · intro x hx
    /-
      case intro.a
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) x
    -/
    rw [← (B.dualBasis hB b).sum_repr x]
    /-
      case intro.a
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) (Finset.un …
    -/
    apply sum_mem
    /-
      case intro.a.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      ⊢ ∀ (c : ι), Membership.mem Finset.univ c → Membership.mem (Submodule.span R ( …
    -/
    rintro i -
    /-
      case intro.a.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      i : ι
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) (HSMul.hSM …
    -/
    obtain ⟨r, hr⟩ := Submodule.mem_one.mp <| hx (b i) (Submodule.subset_span ⟨_, rfl⟩)
    /-
      case intro.a.h.intro
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      i : ι
      r : R
      hr : Eq ((algebraMap R S) r) ((B x) (b i))
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) (HSMul.hSM …
    -/
    simp only [dualBasis_repr_apply, ← hr, Algebra.linearMap_apply, algebraMap_smul]
    /-
      case intro.a.h.intro
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      i : ι
      r : R
      hr : Eq ((algebraMap R S) r) ((B x) (b i))
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) (HSMul.hSM …
    -/
    apply Submodule.smul_mem
    /-
      case intro.a.h.intro.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      x : M
      hx : Membership.mem (B.dualSubmodule (Submodule.span R (Set.range ⇑b))) x
      i : ι
      r : R
      hr : Eq ((algebraMap R S) r) ((B x) (b i))
      ⊢ Membership.mem (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) ((B.dualBa …
    -/
    exact Submodule.subset_span ⟨_, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case intro.a
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      ⊢ LE.le (Submodule.span R (Set.range ⇑(B.dualBasis hB b))) (B.dualSubmodule (S …
    -/
  · rw [Submodule.span_le]
    /-
      case intro.a
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      ⊢ HasSubset.Subset (Set.range ⇑(B.dualBasis hB b)) ↑(B.dualSubmodule (Submodul …
    -/
    rintro _ ⟨i, rfl⟩ y hy
    /-
      case intro.a.intro
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      y : M
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) y
      ⊢ Membership.mem 1 ((B ((B.dualBasis hB b) i)) y)
    -/
    obtain ⟨f, rfl⟩ := (mem_span_range_iff_exists_fun _).mp hy
    /-
      case intro.a.intro.intro
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      ⊢ Membership.mem 1 ((B ((B.dualBasis hB b) i)) (Finset.univ.sum fun i => HSMul …
    -/
    simp only [map_sum, map_smul]
    /-
      case intro.a.intro.intro
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      ⊢ Membership.mem 1 (Finset.univ.sum fun x => (B ((B.dualBasis hB b) i)) (HSMul …
    -/
    apply sum_mem
    /-
      case intro.a.intro.intro.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      ⊢ ∀ (c : ι), Membership.mem Finset.univ c → Membership.mem 1 ((B ((B.dualBasis …
    -/
    rintro j -
    /-
      case intro.a.intro.intro.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      j : ι
      ⊢ Membership.mem 1 ((B ((B.dualBasis hB b) i)) (HSMul.hSMul (f j) (b j)))
    -/
    rw [← IsScalarTower.algebraMap_smul S (f j), map_smul]
    /-
      case intro.a.intro.intro.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      j : ι
      ⊢ Membership.mem 1 (HSMul.hSMul ((algebraMap R S) (f j)) ((B ((B.dualBasis hB  …
    -/
    simp_rw [apply_dualBasis_left]
    /-
      case intro.a.intro.intro.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      j : ι
      ⊢ Membership.mem 1 (HSMul.hSMul ((algebraMap R S) (f j)) (ite (Eq j i) 1 0))
    -/
    rw [smul_eq_mul, mul_ite, mul_one, mul_zero, ← (algebraMap R S).map_zero, ← apply_ite]
    /-
      case intro.a.intro.intro.h
      R : Type u_4
      S : Type u_2
      M : Type u_3
      inst✝⁸ : CommRing R
      inst✝⁷ : Field S
      inst✝⁶ : AddCommGroup M
      inst✝⁵ : Algebra R S
      inst✝⁴ : Module R M
      inst✝³ : Module S M
      inst✝² : IsScalarTower R S M
      B : LinearMap.BilinForm S M
      ι : Type u_1
      inst✝¹ : Finite ι
      inst✝ : DecidableEq ι
      hB : B.Nondegenerate
      b : Basis ι S M
      val✝ : Fintype ι
      i : ι
      f : ι → R
      hy : Membership.mem (Submodule.span R (Set.range ⇑b)) (Finset.univ.sum fun i = …
      j : ι
      ⊢ Membership.mem 1 ((algebraMap R S) (ite (Eq j i) (f j) 0))
    -/
    exact Submodule.mem_one.mpr ⟨_, rfl⟩
    /-
      🎉 no goals
    -/


lemma dualSubmodule_dualSubmodule_flip_of_basis {ι : Type*} [Finite ι]
    (hB : B.Nondegenerate) (b : Basis ι S M) :
    B.dualSubmodule (B.flip.dualSubmodule (Submodule.span R (Set.range b))) =
      Submodule.span R (Set.range b) := by
  classical
  letI := FiniteDimensional.of_fintype_basis b
  rw [dualSubmodule_span_of_basis _ hB.flip, dualSubmodule_span_of_basis B hB,
    dualBasis_dualBasis_flip B hB]


lemma dualSubmodule_flip_dualSubmodule_of_basis {ι : Type*} [Finite ι]
    (hB : B.Nondegenerate) (b : Basis ι S M) :
    B.flip.dualSubmodule (B.dualSubmodule (Submodule.span R (Set.range b))) =
      Submodule.span R (Set.range b) := by
  classical
  letI := FiniteDimensional.of_fintype_basis b
  rw [dualSubmodule_span_of_basis B hB, dualSubmodule_span_of_basis _ hB.flip,
    dualBasis_flip_dualBasis B hB]


lemma dualSubmodule_dualSubmodule_of_basis
    {ι} [Finite ι] (hB : B.Nondegenerate) (hB' : B.IsSymm) (b : Basis ι S M) :
    B.dualSubmodule (B.dualSubmodule (Submodule.span R (Set.range b))) =
      Submodule.span R (Set.range b) := by
  classical
  letI := FiniteDimensional.of_fintype_basis b
  rw [dualSubmodule_span_of_basis B hB, dualSubmodule_span_of_basis B hB,
    dualBasis_dualBasis B hB hB']


