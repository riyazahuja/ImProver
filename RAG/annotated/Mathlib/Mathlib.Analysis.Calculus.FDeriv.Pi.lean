@[fun_prop]
theorem hasFDerivAt_update (x : ∀ i, E i) {i : ι} (y : E i) :
    HasFDerivAt (Function.update x i) (.pi (Pi.single i (.id 𝕜 (E i)))) y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type u_3
    inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : (i : ι) → E i
    i : ι
    y : E i
    ⊢ HasFDerivAt (Function.update x i) (ContinuousLinearMap.pi (Pi.single i (Cont …
  -/
  set l := (ContinuousLinearMap.pi (Pi.single i (.id 𝕜 (E i))))
  have update_eq : Function.update x i = (fun _ ↦ x) + l ∘ (· - x i) := by
    ext t j
    dsimp [l, Pi.single, Function.update]
    split_ifs with hji
    · subst hji
      simp
    · simp
  /-
    𝕜 : Type u_1
    ι : Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type u_3
    inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : (i : ι) → E i
    i : ι
    y : E i
    l : ContinuousLinearMap (RingHom.id 𝕜) (E i) ((i : ι) → E i) := ContinuousLine …
    update_eq : Eq (Function.update x i) (HAdd.hAdd (fun x_1 => x) (Function.comp  …
    ⊢ HasFDerivAt (Function.update x i) l y
  -/
  rw [update_eq]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type u_3
    inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : (i : ι) → E i
    i : ι
    y : E i
    l : ContinuousLinearMap (RingHom.id 𝕜) (E i) ((i : ι) → E i) := ContinuousLine …
    update_eq : Eq (Function.update x i) (HAdd.hAdd (fun x_1 => x) (Function.comp  …
    ⊢ HasFDerivAt (HAdd.hAdd (fun x_1 => x) (Function.comp ⇑l fun x_1 => HSub.hSub …
  -/
  convert (hasFDerivAt_const _ _).add (l.hasFDerivAt.comp y (hasFDerivAt_sub_const (x i)))
  /-
    case h.e'_12.h.h
    𝕜 : Type u_1
    ι : Type u_2
    inst✝⁴ : DecidableEq ι
    inst✝³ : Fintype ι
    inst✝² : NontriviallyNormedField 𝕜
    E : ι → Type u_3
    inst✝¹ : (i : ι) → NormedAddCommGroup (E i)
    inst✝ : (i : ι) → NormedSpace 𝕜 (E i)
    x : (i : ι) → E i
    i : ι
    y : E i
    l : ContinuousLinearMap (RingHom.id 𝕜) (E i) ((i : ι) → E i) := ContinuousLine …
    update_eq : Eq (Function.update x i) (HAdd.hAdd (fun x_1 => x) (Function.comp  …
    e_8✝ : Eq Pi.addCommGroup SeminormedAddCommGroup.toAddCommGroup
    he✝ : Eq (Pi.module ι E 𝕜) NormedSpace.toModule
    e_10✝ : Eq Pi.topologicalSpace UniformSpace.toTopologicalSpace
    ⊢ Eq l (HAdd.hAdd 0 (l.comp (ContinuousLinearMap.id 𝕜 (E i))))
  -/
  rw [zero_add, ContinuousLinearMap.comp_id]
  /-
    🎉 no goals
  -/


@[fun_prop]
theorem hasFDerivAt_single {i : ι} (y : E i) :
    HasFDerivAt (Pi.single i) (.pi (Pi.single i (.id 𝕜 (E i)))) y :=
  hasFDerivAt_update 0 y


theorem fderiv_update (x : ∀ i, E i) {i : ι} (y : E i) :
    fderiv 𝕜 (Function.update x i) y = .pi (Pi.single i (.id 𝕜 (E i))) :=
  (hasFDerivAt_update x y).fderiv


theorem fderiv_single {i : ι} (y : E i) :
    fderiv 𝕜 (Pi.single i) y = .pi (Pi.single i (.id 𝕜 (E i))) :=
  fderiv_update 0 y

