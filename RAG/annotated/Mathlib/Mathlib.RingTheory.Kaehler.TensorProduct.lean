/-- (Implementation). `A`-action on `S ⊗[R] Ω[A⁄R]`. -/
noncomputable
abbrev mulActionBaseChange :
  MulAction A (S ⊗[R] Ω[A⁄R]) := (TensorProduct.comm R S (Ω[A⁄R])).toEquiv.mulAction A


@[simp]
lemma mulActionBaseChange_smul_tmul (a : A) (s : S) (x : Ω[A⁄R]) :
    a • (s ⊗ₜ[R] x) = s ⊗ₜ (a • x) := rfl


@[local simp]
lemma mulActionBaseChange_smul_zero (a : A) :
    a • (0 : S ⊗[R] Ω[A⁄R]) = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a : A
    ⊢ Eq (HSMul.hSMul a 0) 0
  -/
  rw [← zero_tmul _ (0 : Ω[A⁄R]), mulActionBaseChange_smul_tmul, smul_zero]
  /-
    🎉 no goals
  -/


@[local simp]
lemma mulActionBaseChange_smul_add (a : A) (x y : S ⊗[R] Ω[A⁄R]) :
    a • (x + y) = a • x + a • y := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a : A
    x y : TensorProduct R S (KaehlerDifferential R A)
    ⊢ Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
  -/
  show (TensorProduct.comm R S (Ω[A⁄R])).symm (a • (TensorProduct.comm R S (Ω[A⁄R])) (x + y)) = _
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a : A
    x y : TensorProduct R S (KaehlerDifferential R A)
    ⊢ Eq ((TensorProduct.comm R S (KaehlerDifferential R A)).symm (HSMul.hSMul a ( …
  -/
  rw [map_add, smul_add, map_add]
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : Algebra R S
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    a : A
    x y : TensorProduct R S (KaehlerDifferential R A)
    ⊢ Eq (HAdd.hAdd ((TensorProduct.comm R S (KaehlerDifferential R A)).symm (HSMu …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- (Implementation). `A`-module structure on `S ⊗[R] Ω[A⁄R]`. -/
noncomputable
abbrev moduleBaseChange :
    Module A (S ⊗[R] Ω[A⁄R]) where
  __ := (TensorProduct.comm R S (Ω[A⁄R])).toEquiv.mulAction A
                       /-
                         R : Type u_1
                         S : Type u_2
                         A : Type u_3
                         B : Type u_4
                         inst✝¹⁰ : CommRing R
                         inst✝⁹ : CommRing S
                         inst✝⁸ : Algebra R S
                         inst✝⁷ : CommRing A
                         inst✝⁶ : CommRing B
                         inst✝⁵ : Algebra R A
                         inst✝⁴ : Algebra R B
                         inst✝³ : Algebra A B
                         inst✝² : Algebra S B
                         inst✝¹ : IsScalarTower R A B
                         inst✝ : IsScalarTower R S B
                         r s : A
                         x : TensorProduct R S (KaehlerDifferential R A)
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd r s) x) (HAdd.hAdd (HSMul.hSMul r x) (HSMul.hSMul …
                       -/
                                       /-
                                         🎉 no goals
                                       -/
                  /-
                    R : Type u_1
                    S : Type u_2
                    A : Type u_3
                    B : Type u_4
                    inst✝¹⁰ : CommRing R
                    inst✝⁹ : CommRing S
                    inst✝⁸ : Algebra R S
                    inst✝⁷ : CommRing A
                    inst✝⁶ : CommRing B
                    inst✝⁵ : Algebra R A
                    inst✝⁴ : Algebra R B
                    inst✝³ : Algebra A B
                    inst✝² : Algebra S B
                    inst✝¹ : IsScalarTower R A B
                    inst✝ : IsScalarTower R S B
                    ⊢ ∀ (a : A), Eq (HSMul.hSMul a 0) 0
                  -/
                                       /-
                                         🎉 no goals
                                       -/
                  /-
                    🎉 no goals
                  -/
                 /-
                   R : Type u_1
                   S : Type u_2
                   A : Type u_3
                   B : Type u_4
                   inst✝¹⁰ : CommRing R
                   inst✝⁹ : CommRing S
                   inst✝⁸ : Algebra R S
                   inst✝⁷ : CommRing A
                   inst✝⁶ : CommRing B
                   inst✝⁵ : Algebra R A
                   inst✝⁴ : Algebra R B
                   inst✝³ : Algebra A B
                   inst✝² : Algebra S B
                   inst✝¹ : IsScalarTower R A B
                   inst✝ : IsScalarTower R S B
                   ⊢ ∀ (a : A) (x y : TensorProduct R S (KaehlerDifferential R A)), Eq (HSMul.hSM …
                 -/
  add_smul r s x := by induction x <;> simp [add_smul, tmul_add, *, add_add_add_comm]
                 /-
                   🎉 no goals
                 -/
                                       /-
                                         🎉 no goals
                                       -/
                    /-
                      R : Type u_1
                      S : Type u_2
                      A : Type u_3
                      B : Type u_4
                      inst✝¹⁰ : CommRing R
                      inst✝⁹ : CommRing S
                      inst✝⁸ : Algebra R S
                      inst✝⁷ : CommRing A
                      inst✝⁶ : CommRing B
                      inst✝⁵ : Algebra R A
                      inst✝⁴ : Algebra R B
                      inst✝³ : Algebra A B
                      inst✝² : Algebra S B
                      inst✝¹ : IsScalarTower R A B
                      inst✝ : IsScalarTower R S B
                      x : TensorProduct R S (KaehlerDifferential R A)
                      ⊢ Eq (HSMul.hSMul 0 x) 0
                    -/
                                    /-
                                      🎉 no goals
                                    -/
                                    /-
                                      🎉 no goals
                                    -/
  zero_smul x := by induction x <;> simp [*]
                                    /-
                                      🎉 no goals
                                    -/
  smul_zero := by simp
  smul_add := by simp


instance : IsScalarTower R A (S ⊗[R] Ω[A⁄R]) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : Algebra S B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    ⊢ IsScalarTower R A (TensorProduct R S (KaehlerDifferential R A))
  -/
  apply IsScalarTower.of_algebraMap_smul
  /-
    case h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : Algebra S B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    ⊢ ∀ (r : R) (x : TensorProduct R S (KaehlerDifferential R A)), Eq (HSMul.hSMul …
  -/
  intro r x
  /-
    case h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : Algebra S B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    r : R
    x : TensorProduct R S (KaehlerDifferential R A)
    ⊢ Eq (HSMul.hSMul ((algebraMap R A) r) x) (HSMul.hSMul r x)
  -/
  induction x
    /-
      case h.zero
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      r : R
      ⊢ Eq (HSMul.hSMul ((algebraMap R A) r) 0) (HSMul.hSMul r 0)
    -/
  · simp only [smul_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.tmul
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      r : R
      x✝ : S
      y✝ : KaehlerDifferential R A
      ⊢ Eq (HSMul.hSMul ((algebraMap R A) r) (TensorProduct.tmul R x✝ y✝)) (HSMul.hS …
    -/
  · rw [mulActionBaseChange_smul_tmul, algebraMap_smul, tmul_smul]
    /-
      🎉 no goals
    -/
    /-
      case h.add
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      r : R
      x✝ y✝ : TensorProduct R S (KaehlerDifferential R A)
      a✝¹ : Eq (HSMul.hSMul ((algebraMap R A) r) x✝) (HSMul.hSMul r x✝)
      a✝ : Eq (HSMul.hSMul ((algebraMap R A) r) y✝) (HSMul.hSMul r y✝)
      ⊢ Eq (HSMul.hSMul ((algebraMap R A) r) (HAdd.hAdd x✝ y✝)) (HSMul.hSMul r (HAdd …
    -/
  · simp only [smul_add, *]
    /-
      🎉 no goals
    -/


instance : SMulCommClass S A (S ⊗[R] Ω[A⁄R]) where
  smul_comm s a x := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      s : S
      a : A
      x : TensorProduct R S (KaehlerDifferential R A)
      ⊢ Eq (HSMul.hSMul s (HSMul.hSMul a x)) (HSMul.hSMul a (HSMul.hSMul s x))
    -/
    induction x
      /-
        case zero
        R : Type u_1
        S : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹⁰ : CommRing R
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        inst✝⁷ : CommRing A
        inst✝⁶ : CommRing B
        inst✝⁵ : Algebra R A
        inst✝⁴ : Algebra R B
        inst✝³ : Algebra A B
        inst✝² : Algebra S B
        inst✝¹ : IsScalarTower R A B
        inst✝ : IsScalarTower R S B
        s : S
        a : A
        ⊢ Eq (HSMul.hSMul s (HSMul.hSMul a 0)) (HSMul.hSMul a (HSMul.hSMul s 0))
      -/
    · simp only [smul_zero]
      /-
        🎉 no goals
      -/
      /-
        case tmul
        R : Type u_1
        S : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹⁰ : CommRing R
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        inst✝⁷ : CommRing A
        inst✝⁶ : CommRing B
        inst✝⁵ : Algebra R A
        inst✝⁴ : Algebra R B
        inst✝³ : Algebra A B
        inst✝² : Algebra S B
        inst✝¹ : IsScalarTower R A B
        inst✝ : IsScalarTower R S B
        s : S
        a : A
        x✝ : S
        y✝ : KaehlerDifferential R A
        ⊢ Eq (HSMul.hSMul s (HSMul.hSMul a (TensorProduct.tmul R x✝ y✝))) (HSMul.hSMul …
      -/
    · rw [mulActionBaseChange_smul_tmul, smul_tmul', smul_tmul', mulActionBaseChange_smul_tmul]
      /-
        🎉 no goals
      -/
      /-
        case add
        R : Type u_1
        S : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝¹⁰ : CommRing R
        inst✝⁹ : CommRing S
        inst✝⁸ : Algebra R S
        inst✝⁷ : CommRing A
        inst✝⁶ : CommRing B
        inst✝⁵ : Algebra R A
        inst✝⁴ : Algebra R B
        inst✝³ : Algebra A B
        inst✝² : Algebra S B
        inst✝¹ : IsScalarTower R A B
        inst✝ : IsScalarTower R S B
        s : S
        a : A
        x✝ y✝ : TensorProduct R S (KaehlerDifferential R A)
        a✝¹ : Eq (HSMul.hSMul s (HSMul.hSMul a x✝)) (HSMul.hSMul a (HSMul.hSMul s x✝))
        a✝ : Eq (HSMul.hSMul s (HSMul.hSMul a y✝)) (HSMul.hSMul a (HSMul.hSMul s y✝))
        ⊢ Eq (HSMul.hSMul s (HSMul.hSMul a (HAdd.hAdd x✝ y✝))) (HSMul.hSMul a (HSMul.h …
      -/
    · simp only [smul_add, *]
      /-
        🎉 no goals
      -/


instance : SMulCommClass A S (S ⊗[R] Ω[A⁄R]) where
                        /-
                          R : Type u_1
                          S : Type u_2
                          A : Type u_3
                          B : Type u_4
                          inst✝¹⁰ : CommRing R
                          inst✝⁹ : CommRing S
                          inst✝⁸ : Algebra R S
                          inst✝⁷ : CommRing A
                          inst✝⁶ : CommRing B
                          inst✝⁵ : Algebra R A
                          inst✝⁴ : Algebra R B
                          inst✝³ : Algebra A B
                          inst✝² : Algebra S B
                          inst✝¹ : IsScalarTower R A B
                          inst✝ : IsScalarTower R S B
                          s : A
                          a : S
                          x : TensorProduct R S (KaehlerDifferential R A)
                          ⊢ Eq (HSMul.hSMul s (HSMul.hSMul a x)) (HSMul.hSMul a (HSMul.hSMul s x))
                        -/
  smul_comm s a x := by rw [← smul_comm]
                        /-
                          🎉 no goals
                        -/


/-- (Implementation). `B = S ⊗[R] A`-module structure on `S ⊗[R] Ω[A⁄R]`. -/
@[reducible] noncomputable
def moduleBaseChange' [Algebra.IsPushout R S A B] :
    Module B (S ⊗[R] Ω[A⁄R]) :=
  Module.compHom _ (Algebra.pushoutDesc B (Algebra.lsmul R (A := S) S (S ⊗[R] Ω[A⁄R]))
    (Algebra.lsmul R (A := A) _ _) (LinearMap.ext <| smul_comm · ·)).toRingHom


instance [Algebra.IsPushout R S A B] :
    IsScalarTower A B (S ⊗[R] Ω[A⁄R]) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ IsScalarTower A B (TensorProduct R S (KaehlerDifferential R A))
  -/
  apply IsScalarTower.of_algebraMap_smul
  /-
    case h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ ∀ (r : A) (x : TensorProduct R S (KaehlerDifferential R A)), Eq (HSMul.hSMul …
  -/
  intro r x
  show (Algebra.pushoutDesc B (Algebra.lsmul R (A := S) S (S ⊗[R] Ω[A⁄R]))
    (Algebra.lsmul R (A := A) _ _) (LinearMap.ext <| smul_comm · ·)
      (algebraMap A B r)) • x = r • x
  /-
    case h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    r : A
    x : TensorProduct R S (KaehlerDifferential R A)
    ⊢ Eq (HSMul.hSMul ((Algebra.pushoutDesc B (Algebra.lsmul R S (TensorProduct R  …
  -/
  simp only [Algebra.pushoutDesc_right, LinearMap.smul_def, Algebra.lsmul_coe]
  /-
    🎉 no goals
  -/


instance [Algebra.IsPushout R S A B] :
    IsScalarTower S B (S ⊗[R] Ω[A⁄R]) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ IsScalarTower S B (TensorProduct R S (KaehlerDifferential R A))
  -/
  apply IsScalarTower.of_algebraMap_smul
  /-
    case h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ ∀ (r : S) (x : TensorProduct R S (KaehlerDifferential R A)), Eq (HSMul.hSMul …
  -/
  intro r x
  show (Algebra.pushoutDesc B (Algebra.lsmul R (A := S) S (S ⊗[R] Ω[A⁄R]))
    (Algebra.lsmul R (A := A) _ _) (LinearMap.ext <| smul_comm · ·)
      (algebraMap S B r)) • x = r • x
  /-
    case h
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    r : S
    x : TensorProduct R S (KaehlerDifferential R A)
    ⊢ Eq (HSMul.hSMul ((Algebra.pushoutDesc B (Algebra.lsmul R S (TensorProduct R  …
  -/
  simp only [Algebra.pushoutDesc_left, LinearMap.smul_def, Algebra.lsmul_coe]
  /-
    🎉 no goals
  -/


lemma map_liftBaseChange_smul [h : Algebra.IsPushout R S A B] (b : B) (x) :
    ((map R S A B).restrictScalars R).liftBaseChange S (b • x) =
    b • ((map R S A B).restrictScalars R).liftBaseChange S x := by
  induction b using h.1.inductionOn with
  | h₁ => simp only [zero_smul, map_zero]
  | h₃ s b e => rw [smul_assoc, map_smul, e, smul_assoc]
  | h₄ b₁ b₂ e₁ e₂ => simp only [map_add, e₁, e₂, add_smul]
  | h₂ a =>
    induction x
    · simp only [smul_zero, map_zero]
    · simp [smul_comm]
    · simp only [map_add, smul_add, *]


/-- (Implementation).
The `S`-derivation `B = S ⊗[R] A` to `S ⊗[R] Ω[A⁄R]` sending `a ⊗ b` to `a ⊗ d b`. -/
noncomputable
def derivationTensorProduct [h : Algebra.IsPushout R S A B] :
    Derivation S B (S ⊗[R] Ω[A⁄R]) where
  __ := h.out.lift ((TensorProduct.mk R S (Ω[A⁄R]) 1).comp (D R A).toLinearMap)
  map_one_eq_zero' := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      ⊢ Eq (__spread✝⁻⁰ 1) 0
    -/
    rw [← (algebraMap A B).map_one]
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      ⊢ Eq (__spread✝⁻⁰ ((algebraMap A B) 1)) 0
    -/
    refine (h.out.lift_eq _ _).trans ?_
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      ⊢ Eq ((((TensorProduct.mk R S (KaehlerDifferential R A)) 1).comp ↑(KaehlerDiff …
    -/
    dsimp
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      ⊢ Eq (TensorProduct.tmul R 1 ((KaehlerDifferential.D R A) 1)) 0
    -/
    rw [Derivation.map_one_eq_zero, TensorProduct.tmul_zero]
    /-
      🎉 no goals
    -/
  leibniz' a b := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      a b : B
      ⊢ Eq (__spread✝⁻⁰ (HMul.hMul a b)) (HAdd.hAdd (HSMul.hSMul a (__spread✝⁻⁰ b))  …
    -/
    dsimp
    induction a using h.out.inductionOn with
    | h₁ => rw [map_zero, zero_smul, smul_zero, zero_add, zero_mul, map_zero]
    | h₃ x y e =>
      rw [smul_mul_assoc, map_smul, e, map_smul, smul_add,
        smul_comm x b, smul_assoc]
    | h₄ b₁ b₂ e₁ e₂ => simp only [add_mul, add_smul, map_add, e₁, e₂, smul_add, add_add_add_comm]
    | h₂ z =>
      dsimp
      induction b using h.out.inductionOn with
      | h₁ => rw [map_zero, zero_smul, smul_zero, zero_add, mul_zero, map_zero]
      | h₂ =>
        simp only [AlgHom.toLinearMap_apply, IsScalarTower.coe_toAlgHom',
          algebraMap_smul, ← map_mul]
        erw [h.out.lift_eq, h.out.lift_eq, h.out.lift_eq]
        simp only [LinearMap.coe_comp, Derivation.coeFn_coe, Function.comp_apply,
          Derivation.leibniz, mk_apply, mulActionBaseChange_smul_tmul, TensorProduct.tmul_add]
      | h₃ _ _ e =>
        rw [mul_comm, smul_mul_assoc, map_smul, mul_comm, e,
            map_smul, smul_add, smul_comm, smul_assoc]
      | h₄ _ _ e₁ e₂ => simp only [mul_add, add_smul, map_add, e₁, e₂, smul_add, add_add_add_comm]


lemma derivationTensorProduct_algebraMap [Algebra.IsPushout R S A B] (x) :
    derivationTensorProduct R S A B (algebraMap A B x) =
    1 ⊗ₜ D _ _ x :=
IsBaseChange.lift_eq _ _ _


lemma tensorKaehlerEquiv_left_inv [Algebra.IsPushout R S A B] :
    ((derivationTensorProduct R S A B).liftKaehlerDifferential.restrictScalars S).comp
    (((map R S A B).restrictScalars R).liftBaseChange S) = LinearMap.id := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ Eq ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKaehlerDif …
  -/
  refine LinearMap.restrictScalars_injective R ?_
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ Eq (↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKaehle …
  -/
  apply TensorProduct.ext'
  /-
    case H
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    ⊢ ∀ (x : S) (y : KaehlerDifferential R A), Eq ((↑R ((↑S (KaehlerDifferential.d …
  -/
  intro x y
  /-
    case H
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    x : S
    y : KaehlerDifferential R A
    ⊢ Eq ((↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKaehl …
  -/
  obtain ⟨y, rfl⟩ := tensorProductTo_surjective _ _ y
  /-
    case H.intro
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹¹ : CommRing R
    inst✝¹⁰ : CommRing S
    inst✝⁹ : Algebra R S
    inst✝⁸ : CommRing A
    inst✝⁷ : CommRing B
    inst✝⁶ : Algebra R A
    inst✝⁵ : Algebra R B
    inst✝⁴ : Algebra A B
    inst✝³ : Algebra S B
    inst✝² : IsScalarTower R A B
    inst✝¹ : IsScalarTower R S B
    inst✝ : Algebra.IsPushout R S A B
    x : S
    y : TensorProduct R A A
    ⊢ Eq ((↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKaehl …
  -/
  induction y
    /-
      case H.intro.zero
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra S B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : Algebra.IsPushout R S A B
      x : S
      ⊢ Eq ((↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKaehl …
    -/
  · simp only [map_zero, TensorProduct.tmul_zero]
    /-
      🎉 no goals
    -/
  · simp only [LinearMap.restrictScalars_comp, Derivation.tensorProductTo_tmul, LinearMap.coe_comp,
      LinearMap.coe_restrictScalars, Function.comp_apply, LinearMap.liftBaseChange_tmul, map_smul,
      map_D, LinearMap.map_smul_of_tower, Derivation.liftKaehlerDifferential_comp_D,
      LinearMap.id_coe, id_eq, derivationTensorProduct_algebraMap]
    /-
      case H.intro.tmul
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra S B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : Algebra.IsPushout R S A B
      x : S
      x✝ y✝ : A
      ⊢ Eq (HSMul.hSMul x (HSMul.hSMul x✝ (TensorProduct.tmul R 1 ((KaehlerDifferent …
    -/
    rw [smul_comm, TensorProduct.smul_tmul', smul_eq_mul, mul_one]
    /-
      case H.intro.tmul
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra S B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : Algebra.IsPushout R S A B
      x : S
      x✝ y✝ : A
      ⊢ Eq (HSMul.hSMul x✝ (TensorProduct.tmul R x ((KaehlerDifferential.D R A) y✝)) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case H.intro.add
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹¹ : CommRing R
      inst✝¹⁰ : CommRing S
      inst✝⁹ : Algebra R S
      inst✝⁸ : CommRing A
      inst✝⁷ : CommRing B
      inst✝⁶ : Algebra R A
      inst✝⁵ : Algebra R B
      inst✝⁴ : Algebra A B
      inst✝³ : Algebra S B
      inst✝² : IsScalarTower R A B
      inst✝¹ : IsScalarTower R S B
      inst✝ : Algebra.IsPushout R S A B
      x : S
      x✝ y✝ : TensorProduct R A A
      a✝¹ : Eq ((↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftK …
      a✝ : Eq ((↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKa …
      ⊢ Eq ((↑R ((↑S (KaehlerDifferential.derivationTensorProduct R S A B).liftKaehl …
    -/
  · simp only [map_add, TensorProduct.tmul_add, *]
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism `(S ⊗[R] Ω[A⁄R]) ≃ₗ[S] Ω[B⁄S]` for `B = S ⊗[R] A`. -/
@[simps! symm_apply] noncomputable
def tensorKaehlerEquiv [h : Algebra.IsPushout R S A B] :
    (S ⊗[R] Ω[A⁄R]) ≃ₗ[S] Ω[B⁄S] where
  __ := ((map R S A B).restrictScalars R).liftBaseChange S
  invFun := (derivationTensorProduct R S A B).liftKaehlerDifferential
  left_inv := LinearMap.congr_fun (tensorKaehlerEquiv_left_inv R S A B)
  right_inv x := by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      x : KaehlerDifferential S B
      ⊢ Eq (__spread✝⁻⁰.toFun ((KaehlerDifferential.derivationTensorProduct R S A B) …
    -/
    obtain ⟨x, rfl⟩ := tensorProductTo_surjective _ _ x
    /-
      case intro
      R : Type u_1
      S : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝¹⁰ : CommRing R
      inst✝⁹ : CommRing S
      inst✝⁸ : Algebra R S
      inst✝⁷ : CommRing A
      inst✝⁶ : CommRing B
      inst✝⁵ : Algebra R A
      inst✝⁴ : Algebra R B
      inst✝³ : Algebra A B
      inst✝² : Algebra S B
      inst✝¹ : IsScalarTower R A B
      inst✝ : IsScalarTower R S B
      h : Algebra.IsPushout R S A B
      x : TensorProduct S B B
      ⊢ Eq (__spread✝⁻⁰.toFun ((KaehlerDifferential.derivationTensorProduct R S A B) …
    -/
    dsimp
    induction x with
    | zero => simp
    | add x y e₁ e₂ => simp only [map_add, e₁, e₂]
    | tmul x y =>
      dsimp
      simp only [Derivation.tensorProductTo_tmul, LinearMap.map_smul,
        Derivation.liftKaehlerDifferential_comp_D, map_liftBaseChange_smul]
      induction y using h.1.inductionOn
      · simp only [map_zero, smul_zero]
      · simp only [AlgHom.toLinearMap_apply, IsScalarTower.coe_toAlgHom',
          derivationTensorProduct_algebraMap, LinearMap.liftBaseChange_tmul,
          LinearMap.coe_restrictScalars, map_D, one_smul]
      · simp only [Derivation.map_smul, LinearMap.map_smul, *, smul_comm x]
      · simp only [map_add, smul_add, *]


@[simp]
lemma tensorKaehlerEquiv_tmul [Algebra.IsPushout R S A B] (a b) :
    tensorKaehlerEquiv R S A B (a ⊗ₜ b) = a • map R S A B b :=
  LinearMap.liftBaseChange_tmul _ _ _ _


/--
If `B` is the tensor product of `S` and `A` over `R`,
then `Ω[B⁄S]` is the base change of `Ω[A⁄R]` along `R → S`.
-/
lemma isBaseChange [h : Algebra.IsPushout R S A B] :
    IsBaseChange S ((map R S A B).restrictScalars R) := by
  convert (TensorProduct.isBaseChange R (Ω[A⁄R]) S).comp
    (IsBaseChange.ofEquiv (tensorKaehlerEquiv R S A B))
  /-
    case h.e'_14
    R : Type u_1
    S : Type u_2
    A : Type u_3
    B : Type u_4
    inst✝¹⁰ : CommRing R
    inst✝⁹ : CommRing S
    inst✝⁸ : Algebra R S
    inst✝⁷ : CommRing A
    inst✝⁶ : CommRing B
    inst✝⁵ : Algebra R A
    inst✝⁴ : Algebra R B
    inst✝³ : Algebra A B
    inst✝² : Algebra S B
    inst✝¹ : IsScalarTower R A B
    inst✝ : IsScalarTower R S B
    h : Algebra.IsPushout R S A B
    ⊢ Eq (↑R (KaehlerDifferential.map R S A B)) ((↑R ↑(KaehlerDifferential.tensorK …
  -/
  refine LinearMap.ext fun x ↦ ?_
  simp only [LinearMap.coe_restrictScalars, LinearMap.coe_comp, LinearEquiv.coe_coe,
    Function.comp_apply, mk_apply, tensorKaehlerEquiv_tmul, one_smul]


instance isLocalizedModule (p : Submonoid R) [IsLocalization p S]
      [IsLocalization (Algebra.algebraMapSubmonoid A p) B] :
    IsLocalizedModule p ((map R S A B).restrictScalars R) :=
  have := (Algebra.isPushout_of_isLocalization p S A B).symm
  (isLocalizedModule_iff_isBaseChange p S _).mpr (isBaseChange R S A B)


instance isLocalizedModule_of_isLocalizedModule (p : Submonoid R) [IsLocalization p S]
      [IsLocalizedModule p (IsScalarTower.toAlgHom R A B).toLinearMap] :
    IsLocalizedModule p ((map R S A B).restrictScalars R) :=
  have : IsLocalization (Algebra.algebraMapSubmonoid A p) B :=
    isLocalizedModule_iff_isLocalization.mp inferInstance
  inferInstance


