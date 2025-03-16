local notation "∞" => (⊤ : ℕ∞)

-- the following two instances prevent poorly understood type class inference timeout problems

                                                               /-
                                                                 𝕜 : Type u_1
                                                                 inst✝⁵ : NontriviallyNormedField 𝕜
                                                                 E : Type u_2
                                                                 inst✝⁴ : NormedAddCommGroup E
                                                                 inst✝³ : NormedSpace 𝕜 E
                                                                 H : Type u_3
                                                                 inst✝² : TopologicalSpace H
                                                                 I : ModelWithCorners 𝕜 E H
                                                                 M : Type u_4
                                                                 inst✝¹ : TopologicalSpace M
                                                                 inst✝ : ChartedSpace H M
                                                                 n : ENat
                                                                 ⊢ Algebra 𝕜 (ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top)
                                                               -/
instance smoothFunctionsAlgebra : Algebra 𝕜 C^∞⟮I, M; 𝕜⟯ := by infer_instance
                                                               /-
                                                                 🎉 no goals
                                                               -/


                                                                                  /-
                                                                                    𝕜 : Type u_1
                                                                                    inst✝⁵ : NontriviallyNormedField 𝕜
                                                                                    E : Type u_2
                                                                                    inst✝⁴ : NormedAddCommGroup E
                                                                                    inst✝³ : NormedSpace 𝕜 E
                                                                                    H : Type u_3
                                                                                    inst✝² : TopologicalSpace H
                                                                                    I : ModelWithCorners 𝕜 E H
                                                                                    M : Type u_4
                                                                                    inst✝¹ : TopologicalSpace M
                                                                                    inst✝ : ChartedSpace H M
                                                                                    n : ENat
                                                                                    ⊢ IsScalarTower 𝕜 (ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top) (Con …
                                                                                  -/
instance smooth_functions_tower : IsScalarTower 𝕜 C^∞⟮I, M; 𝕜⟯ C^∞⟮I, M; 𝕜⟯ := by infer_instance
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


/-- Type synonym, introduced to put a different `SMul` action on `C^n⟮I, M; 𝕜⟯`
which is defined as `f • r = f(x) * r`. -/
@[nolint unusedArguments]
def PointedSmoothMap (_ : M) :=
  C^n⟮I, M; 𝕜⟯


@[inherit_doc]
scoped[Derivation] notation "C^" n "⟮" I ", " M "; " 𝕜 "⟯⟨" x "⟩" => PointedSmoothMap 𝕜 I M n x


instance instFunLike {x : M} : FunLike C^∞⟮I, M; 𝕜⟯⟨x⟩ M 𝕜 :=
  ContMDiffMap.instFunLike


instance {x : M} : CommRing C^∞⟮I, M; 𝕜⟯⟨x⟩ :=
  SmoothMap.commRing


instance {x : M} : Algebra 𝕜 C^∞⟮I, M; 𝕜⟯⟨x⟩ :=
  SmoothMap.algebra


instance {x : M} : Inhabited C^∞⟮I, M; 𝕜⟯⟨x⟩ :=
  ⟨0⟩


instance {x : M} : Algebra C^∞⟮I, M; 𝕜⟯⟨x⟩ C^∞⟮I, M; 𝕜⟯ :=
  Algebra.id C^∞⟮I, M; 𝕜⟯


instance {x : M} : IsScalarTower 𝕜 C^∞⟮I, M; 𝕜⟯⟨x⟩ C^∞⟮I, M; 𝕜⟯ :=
  IsScalarTower.right


/-- `SmoothMap.evalRingHom` gives rise to an algebra structure of `C^∞⟮I, M; 𝕜⟯` on `𝕜`. -/
instance evalAlgebra {x : M} : Algebra C^∞⟮I, M; 𝕜⟯⟨x⟩ 𝕜 :=
  (SmoothMap.evalRingHom x : C^∞⟮I, M; 𝕜⟯⟨x⟩ →+* 𝕜).toAlgebra


/-- With the `evalAlgebra` algebra structure evaluation is actually an algebra morphism. -/
def eval (x : M) : C^∞⟮I, M; 𝕜⟯ →ₐ[C^∞⟮I, M; 𝕜⟯⟨x⟩] 𝕜 :=
  Algebra.ofId C^∞⟮I, M; 𝕜⟯⟨x⟩ 𝕜


theorem smul_def (x : M) (f : C^∞⟮I, M; 𝕜⟯⟨x⟩) (k : 𝕜) : f • k = f x * k :=
  rfl


instance (x : M) : IsScalarTower 𝕜 C^∞⟮I, M; 𝕜⟯⟨x⟩ 𝕜 where
  smul_assoc k f h := by
    /-
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      H : Type u_3
      inst✝² : TopologicalSpace H
      I : ModelWithCorners 𝕜 E H
      M : Type u_4
      inst✝¹ : TopologicalSpace M
      inst✝ : ChartedSpace H M
      n : ENat
      x : M
      k : 𝕜
      f : PointedSmoothMap 𝕜 I M Top.top x
      h : 𝕜
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul k f) h) (HSMul.hSMul k (HSMul.hSMul f h))
    -/
    rw [smul_def, smul_def, SmoothMap.coe_smul, Pi.smul_apply, smul_eq_mul, smul_eq_mul, mul_assoc]
    /-
      🎉 no goals
    -/


/-- The derivations at a point of a manifold. Some regard this as a possible definition of the
tangent space -/
abbrev PointDerivation (x : M) :=
  Derivation 𝕜 C^∞⟮I, M; 𝕜⟯⟨x⟩ 𝕜


/-- Evaluation at a point gives rise to a `C^∞⟮I, M; 𝕜⟯`-linear map between `C^∞⟮I, M; 𝕜⟯` and `𝕜`.
 -/
def SmoothFunction.evalAt (x : M) : C^∞⟮I, M; 𝕜⟯ →ₗ[C^∞⟮I, M; 𝕜⟯⟨x⟩] 𝕜 :=
  (PointedSmoothMap.eval x).toLinearMap


/-- The evaluation at a point as a linear map. -/
def evalAt (x : M) : Derivation 𝕜 C^∞⟮I, M; 𝕜⟯ C^∞⟮I, M; 𝕜⟯ →ₗ[𝕜] PointDerivation I x :=
  (SmoothFunction.evalAt I x).compDer


theorem evalAt_apply (x : M) : evalAt x X f = (X f) x :=
  rfl


/-- The heterogeneous differential as a linear map. Instead of taking a function as an argument this
differential takes `h : f x = y`. It is particularly handy to deal with situations where the points
on where it has to be evaluated are equal but not definitionally equal. -/
def hfdifferential {f : C^∞⟮I, M; I', M'⟯} {x : M} {y : M'} (h : f x = y) :
    PointDerivation I x →ₗ[𝕜] PointDerivation I' y where
  toFun v :=
    Derivation.mk'
      { toFun := fun g => v (g.comp f)
                                   /-
                                     𝕜 : Type u_1
                                     inst✝¹⁰ : NontriviallyNormedField 𝕜
                                     E : Type u_2
                                     inst✝⁹ : NormedAddCommGroup E
                                     inst✝⁸ : NormedSpace 𝕜 E
                                     H : Type u_3
                                     inst✝⁷ : TopologicalSpace H
                                     I : ModelWithCorners 𝕜 E H
                                     M : Type u_4
                                     inst✝⁶ : TopologicalSpace M
                                     inst✝⁵ : ChartedSpace H M
                                     n : ENat
                                     X : Derivation 𝕜 (ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top) (Cont …
                                     f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top
                                     E' : Type u_5
                                     inst✝⁴ : NormedAddCommGroup E'
                                     inst✝³ : NormedSpace 𝕜 E'
                                     H' : Type u_6
                                     inst✝² : TopologicalSpace H'
                                     I' : ModelWithCorners 𝕜 E' H'
                                     M' : Type u_7
                                     inst✝¹ : TopologicalSpace M'
                                     inst✝ : ChartedSpace H' M'
                                     f : ContMDiffMap I I' M M' Top.top
                                     x : M
                                     y : M'
                                     h : Eq (f x) y
                                     v : PointDerivation I x
                                     g g' : PointedSmoothMap 𝕜 I' M' Top.top y
                                     ⊢ Eq ((fun g => v (ContMDiffMap.comp g f)) (HAdd.hAdd g g')) (HAdd.hAdd ((fun  …
                                   -/
        map_add' := fun g g' => by dsimp; rw [SmoothMap.add_comp, Derivation.map_add]
                                          /-
                                            🎉 no goals
                                          -/
        map_smul' := fun k g => by
          /-
            𝕜 : Type u_1
            inst✝¹⁰ : NontriviallyNormedField 𝕜
            E : Type u_2
            inst✝⁹ : NormedAddCommGroup E
            inst✝⁸ : NormedSpace 𝕜 E
            H : Type u_3
            inst✝⁷ : TopologicalSpace H
            I : ModelWithCorners 𝕜 E H
            M : Type u_4
            inst✝⁶ : TopologicalSpace M
            inst✝⁵ : ChartedSpace H M
            n : ENat
            X : Derivation 𝕜 (ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top) (Cont …
            f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top
            E' : Type u_5
            inst✝⁴ : NormedAddCommGroup E'
            inst✝³ : NormedSpace 𝕜 E'
            H' : Type u_6
            inst✝² : TopologicalSpace H'
            I' : ModelWithCorners 𝕜 E' H'
            M' : Type u_7
            inst✝¹ : TopologicalSpace M'
            inst✝ : ChartedSpace H' M'
            f : ContMDiffMap I I' M M' Top.top
            x : M
            y : M'
            h : Eq (f x) y
            v : PointDerivation I x
            k : 𝕜
            g : PointedSmoothMap 𝕜 I' M' Top.top y
            ⊢ Eq ({ toFun := fun g => v (ContMDiffMap.comp g f), map_add' := ⋯ }.toFun (HS …
          -/
          dsimp; rw [SmoothMap.smul_comp, Derivation.map_smul, smul_eq_mul] }
                 /-
                   🎉 no goals
                 -/
      fun g g' => by
        /-
          𝕜 : Type u_1
          inst✝¹⁰ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁹ : NormedAddCommGroup E
          inst✝⁸ : NormedSpace 𝕜 E
          H : Type u_3
          inst✝⁷ : TopologicalSpace H
          I : ModelWithCorners 𝕜 E H
          M : Type u_4
          inst✝⁶ : TopologicalSpace M
          inst✝⁵ : ChartedSpace H M
          n : ENat
          X : Derivation 𝕜 (ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top) (Cont …
          f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top
          E' : Type u_5
          inst✝⁴ : NormedAddCommGroup E'
          inst✝³ : NormedSpace 𝕜 E'
          H' : Type u_6
          inst✝² : TopologicalSpace H'
          I' : ModelWithCorners 𝕜 E' H'
          M' : Type u_7
          inst✝¹ : TopologicalSpace M'
          inst✝ : ChartedSpace H' M'
          f : ContMDiffMap I I' M M' Top.top
          x : M
          y : M'
          h : Eq (f x) y
          v : PointDerivation I x
          g g' : PointedSmoothMap 𝕜 I' M' Top.top y
          ⊢ Eq ({ toFun := fun g => v (ContMDiffMap.comp g f), map_add' := ⋯, map_smul'  …
        -/
        dsimp
        rw [SmoothMap.mul_comp, Derivation.leibniz,
          PointedSmoothMap.smul_def, ContMDiffMap.comp_apply,
          PointedSmoothMap.smul_def, ContMDiffMap.comp_apply, h]
        /-
          𝕜 : Type u_1
          inst✝¹⁰ : NontriviallyNormedField 𝕜
          E : Type u_2
          inst✝⁹ : NormedAddCommGroup E
          inst✝⁸ : NormedSpace 𝕜 E
          H : Type u_3
          inst✝⁷ : TopologicalSpace H
          I : ModelWithCorners 𝕜 E H
          M : Type u_4
          inst✝⁶ : TopologicalSpace M
          inst✝⁵ : ChartedSpace H M
          n : ENat
          X : Derivation 𝕜 (ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top) (Cont …
          f✝ : ContMDiffMap I (modelWithCornersSelf 𝕜 𝕜) M 𝕜 Top.top
          E' : Type u_5
          inst✝⁴ : NormedAddCommGroup E'
          inst✝³ : NormedSpace 𝕜 E'
          H' : Type u_6
          inst✝² : TopologicalSpace H'
          I' : ModelWithCorners 𝕜 E' H'
          M' : Type u_7
          inst✝¹ : TopologicalSpace M'
          inst✝ : ChartedSpace H' M'
          f : ContMDiffMap I I' M M' Top.top
          x : M
          y : M'
          h : Eq (f x) y
          v : PointDerivation I x
          g g' : PointedSmoothMap 𝕜 I' M' Top.top y
          ⊢ Eq (HAdd.hAdd (HMul.hMul (g y) (v (ContMDiffMap.comp g' f))) (HMul.hMul (g'  …
        -/
        norm_cast
        /-
          🎉 no goals
        -/
  map_smul' _ _ := rfl
  map_add' _ _ := rfl


/-- The homogeneous differential as a linear map. -/
def fdifferential (f : C^∞⟮I, M; I', M'⟯) (x : M) :
    PointDerivation I x →ₗ[𝕜] PointDerivation I' (f x) :=
  hfdifferential (rfl : f x = f x)

-- Standard notation for the differential. The abbreviation is `MId`.

scoped[Manifold] notation "𝒅" => fdifferential

-- Standard notation for the differential. The abbreviation is `MId`.

scoped[Manifold] notation "𝒅ₕ" => hfdifferential


@[simp]
theorem fdifferential_apply (f : C^∞⟮I, M; I', M'⟯) {x : M} (v : PointDerivation I x)
    (g : C^∞⟮I', M'; 𝕜⟯) : 𝒅 f x v g = v (g.comp f) :=
  rfl

@[deprecated (since := "2024-11-11")] alias apply_fdifferential := fdifferential_apply


@[simp]
theorem hfdifferential_apply {f : C^∞⟮I, M; I', M'⟯} {x : M} {y : M'} (h : f x = y)
    (v : PointDerivation I x) (g : C^∞⟮I', M'; 𝕜⟯) : 𝒅ₕ h v g = 𝒅 f x v g :=
  rfl

@[deprecated (since := "2024-11-11")] alias apply_hfdifferential := hfdifferential_apply


@[simp]
theorem fdifferential_comp (g : C^∞⟮I', M'; I'', M''⟯) (f : C^∞⟮I, M; I', M'⟯) (x : M) :
    𝒅 (g.comp f) x = (𝒅 g (f x)).comp (𝒅 f x) :=
  rfl


