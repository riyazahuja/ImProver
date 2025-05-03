theorem hasDerivAt_update (x : ι → 𝕜) (i : ι) (y : 𝕜) :
    HasDerivAt (Function.update x i) (Pi.single i (1 : 𝕜)) y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : NontriviallyNormedField 𝕜
    x : ι → 𝕜
    i : ι
    y : 𝕜
    ⊢ HasDerivAt (Function.update x i) (Pi.single i 1) y
  -/
  convert (hasFDerivAt_update x y).hasDerivAt
  /-
    case h.e'_9.h.e
    𝕜 : Type u_1
    ι : Type u_2
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : NontriviallyNormedField 𝕜
    x : ι → 𝕜
    i : ι
    y : 𝕜
    ⊢ Eq (Pi.single i) ⇑(ContinuousLinearMap.pi (Pi.single i (ContinuousLinearMap. …
  -/
  ext z j
  /-
    case h.e'_9.h.e.h.h
    𝕜 : Type u_1
    ι : Type u_2
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : NontriviallyNormedField 𝕜
    x : ι → 𝕜
    i : ι
    y z : 𝕜
    j : ι
    ⊢ Eq (Pi.single i z j) ((ContinuousLinearMap.pi (Pi.single i (ContinuousLinear …
  -/
  rw [Pi.single, Function.update_apply]
  /-
    case h.e'_9.h.e.h.h
    𝕜 : Type u_1
    ι : Type u_2
    inst✝² : DecidableEq ι
    inst✝¹ : Fintype ι
    inst✝ : NontriviallyNormedField 𝕜
    x : ι → 𝕜
    i : ι
    y z : 𝕜
    j : ι
    ⊢ Eq (ite (Eq j i) z (0 j)) ((ContinuousLinearMap.pi (Pi.single i (ContinuousL …
  -/
  split_ifs with h
    /-
      case pos
      𝕜 : Type u_1
      ι : Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      inst✝ : NontriviallyNormedField 𝕜
      x : ι → 𝕜
      i : ι
      y z : 𝕜
      j : ι
      h : Eq j i
      ⊢ Eq z ((ContinuousLinearMap.pi (Pi.single i (ContinuousLinearMap.id 𝕜 𝕜))) z j)
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      ι : Type u_2
      inst✝² : DecidableEq ι
      inst✝¹ : Fintype ι
      inst✝ : NontriviallyNormedField 𝕜
      x : ι → 𝕜
      i : ι
      y z : 𝕜
      j : ι
      h : Not (Eq j i)
      ⊢ Eq (0 j) ((ContinuousLinearMap.pi (Pi.single i (ContinuousLinearMap.id 𝕜 𝕜)) …
    -/
  · simp [Pi.single_eq_of_ne h]
    /-
      🎉 no goals
    -/


theorem hasDerivAt_single (i : ι) (y : 𝕜) :
    HasDerivAt (Pi.single (f := fun _ ↦ 𝕜) i) (Pi.single i (1 : 𝕜)) y :=
  hasDerivAt_update 0 i y


theorem deriv_update (x : ι → 𝕜) (i : ι) (y : 𝕜) :
    deriv (Function.update x i) y = Pi.single i (1 : 𝕜) :=
  (hasDerivAt_update x i y).deriv


theorem deriv_single (i : ι) (y : 𝕜) :
    deriv (Pi.single (f := fun _ ↦ 𝕜) i) y = Pi.single i (1 : 𝕜) :=
  deriv_update 0 i y

