/-- Given a `ContDiff` function over `𝕂` (which is `ℝ` or `ℂ`) with an invertible
derivative at `a`, returns a `PartialHomeomorph` with `to_fun = f` and `a ∈ source`. -/
def toPartialHomeomorph (hf : ContDiffAt 𝕂 n f a) (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a)
    (hn : 1 ≤ n) : PartialHomeomorph E F :=
  (hf.hasStrictFDerivAt' hf' hn).toPartialHomeomorph f


@[simp]
theorem toPartialHomeomorph_coe (hf : ContDiffAt 𝕂 n f a)
    (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a) (hn : 1 ≤ n) :
    (hf.toPartialHomeomorph f hf' hn : E → F) = f :=
  rfl


theorem mem_toPartialHomeomorph_source (hf : ContDiffAt 𝕂 n f a)
    (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a) (hn : 1 ≤ n) :
    a ∈ (hf.toPartialHomeomorph f hf' hn).source :=
  (hf.hasStrictFDerivAt' hf' hn).mem_toPartialHomeomorph_source


theorem image_mem_toPartialHomeomorph_target (hf : ContDiffAt 𝕂 n f a)
    (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a) (hn : 1 ≤ n) :
    f a ∈ (hf.toPartialHomeomorph f hf' hn).target :=
  (hf.hasStrictFDerivAt' hf' hn).image_mem_toPartialHomeomorph_target


/-- Given a `ContDiff` function over `𝕂` (which is `ℝ` or `ℂ`) with an invertible derivative
at `a`, returns a function that is locally inverse to `f`. -/
def localInverse (hf : ContDiffAt 𝕂 n f a) (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a)
    (hn : 1 ≤ n) : F → E :=
  (hf.hasStrictFDerivAt' hf' hn).localInverse f f' a


theorem localInverse_apply_image (hf : ContDiffAt 𝕂 n f a)
    (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a) (hn : 1 ≤ n) : hf.localInverse hf' hn (f a) = a :=
  (hf.hasStrictFDerivAt' hf' hn).localInverse_apply_image


/-- Given a `ContDiff` function over `𝕂` (which is `ℝ` or `ℂ`) with an invertible derivative
at `a`, the inverse function (produced by `ContDiff.toPartialHomeomorph`) is
also `ContDiff`. -/
theorem to_localInverse (hf : ContDiffAt 𝕂 n f a)
    (hf' : HasFDerivAt f (f' : E →L[𝕂] F) a) (hn : 1 ≤ n) :
    ContDiffAt 𝕂 n (hf.localInverse hf' hn) (f a) := by
  /-
    𝕂 : Type u_1
    inst✝⁵ : RCLike 𝕂
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕂 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕂 F
    inst✝ : CompleteSpace E
    f : E → F
    f' : ContinuousLinearEquiv (RingHom.id 𝕂) E F
    a : E
    n : WithTop ENat
    hf : ContDiffAt 𝕂 n f a
    hf' : HasFDerivAt f (↑f') a
    hn : LE.le 1 n
    ⊢ ContDiffAt 𝕂 n (hf.localInverse hf' hn) (f a)
  -/
  have := hf.localInverse_apply_image hf' hn
  apply (hf.toPartialHomeomorph f hf' hn).contDiffAt_symm
    (image_mem_toPartialHomeomorph_target hf hf' hn)
    /-
      case hf₀'
      𝕂 : Type u_1
      inst✝⁵ : RCLike 𝕂
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕂 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕂 F
      inst✝ : CompleteSpace E
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕂) E F
      a : E
      n : WithTop ENat
      hf : ContDiffAt 𝕂 n f a
      hf' : HasFDerivAt f (↑f') a
      hn : LE.le 1 n
      this : Eq (hf.localInverse hf' hn (f a)) a
      ⊢ HasFDerivAt (↑(ContDiffAt.toPartialHomeomorph f hf hf' hn)) (↑?m.42313) (↑(C …
    -/
  · convert hf'
    /-
      🎉 no goals
    -/
    /-
      case hf
      𝕂 : Type u_1
      inst✝⁵ : RCLike 𝕂
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕂 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕂 F
      inst✝ : CompleteSpace E
      f : E → F
      f' : ContinuousLinearEquiv (RingHom.id 𝕂) E F
      a : E
      n : WithTop ENat
      hf : ContDiffAt 𝕂 n f a
      hf' : HasFDerivAt f (↑f') a
      hn : LE.le 1 n
      this : Eq (hf.localInverse hf' hn (f a)) a
      ⊢ ContDiffAt 𝕂 n (↑(ContDiffAt.toPartialHomeomorph f hf hf' hn)) (↑(ContDiffAt …
    -/
  · convert hf
    /-
      🎉 no goals
    -/


