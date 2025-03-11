/-- A dilation is a map that uniformly scales the edistance between any two points. -/
structure Dilation where
  toFun : α → β
  edist_eq' : ∃ r : ℝ≥0, r ≠ 0 ∧ ∀ x y : α, edist (toFun x) (toFun y) = r * edist x y


@[inherit_doc] infixl:25 " →ᵈ " => Dilation


/-- `DilationClass F α β r` states that `F` is a type of `r`-dilations.
You should extend this typeclass when you extend `Dilation`. -/
class DilationClass (F : Type*) (α β : outParam Type*) [PseudoEMetricSpace α] [PseudoEMetricSpace β]
    [FunLike F α β] : Prop where
  edist_eq' : ∀ f : F, ∃ r : ℝ≥0, r ≠ 0 ∧ ∀ x y : α, edist (f x) (f y) = r * edist x y


instance funLike : FunLike (α →ᵈ β) α β where
  coe := toFun
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               F : Type u_4
                               inst✝¹ : PseudoEMetricSpace α
                               inst✝ : PseudoEMetricSpace β
                               f g : Dilation α β
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


instance toDilationClass : DilationClass (α →ᵈ β) α β where
  edist_eq' f := edist_eq' f


@[simp]
theorem toFun_eq_coe {f : α →ᵈ β} : f.toFun = (f : α → β) :=
  rfl


@[simp]
theorem coe_mk (f : α → β) (h) : ⇑(⟨f, h⟩ : α →ᵈ β) = f :=
  rfl


protected theorem congr_fun {f g : α →ᵈ β} (h : f = g) (x : α) : f x = g x :=
  DFunLike.congr_fun h x


protected theorem congr_arg (f : α →ᵈ β) {x y : α} (h : x = y) : f x = f y :=
  DFunLike.congr_arg f h


@[ext]
theorem ext {f g : α →ᵈ β} (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext f g h


@[simp]
theorem mk_coe (f : α →ᵈ β) (h) : Dilation.mk f h = f :=
  ext fun _ => rfl


/-- Copy of a `Dilation` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
@[simps (config := .asFn)]
protected def copy (f : α →ᵈ β) (f' : α → β) (h : f' = ⇑f) : α →ᵈ β where
  toFun := f'
  edist_eq' := h.symm ▸ f.edist_eq'


theorem copy_eq_self (f : α →ᵈ β) {f' : α → β} (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


open Classical in
/-- The ratio of a dilation `f`. If the ratio is undefined (i.e., the distance between any two
points in `α` is either zero or infinity), then we choose one as the ratio. -/
def ratio [DilationClass F α β] (f : F) : ℝ≥0 :=
  if ∀ x y : α, edist x y = 0 ∨ edist x y = ⊤ then 1 else (DilationClass.edist_eq' f).choose


theorem ratio_of_trivial [DilationClass F α β] (f : F)
    (h : ∀ x y : α, edist x y = 0 ∨ edist x y = ∞) : ratio f = 1 :=
  if_pos h


@[nontriviality]
theorem ratio_of_subsingleton [Subsingleton α] [DilationClass F α β] (f : F) : ratio f = 1 :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        F : Type u_4
                        inst✝⁴ : PseudoEMetricSpace α
                        inst✝³ : PseudoEMetricSpace β
                        inst✝² : FunLike F α β
                        inst✝¹ : Subsingleton α
                        inst✝ : DilationClass F α β
                        f : F
                        x y : α
                        ⊢ Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
                      -/
  if_pos fun x y ↦ by simp [Subsingleton.elim x y]
                      /-
                        🎉 no goals
                      -/


theorem ratio_ne_zero [DilationClass F α β] (f : F) : ratio f ≠ 0 := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    ⊢ Ne (Dilation.ratio f) 0
  -/
  rw [ratio]; split_ifs
    /-
      case pos
      α : Type u_1
      β : Type u_2
      F : Type u_4
      inst✝³ : PseudoEMetricSpace α
      inst✝² : PseudoEMetricSpace β
      inst✝¹ : FunLike F α β
      inst✝ : DilationClass F α β
      f : F
      h✝ : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      ⊢ Ne 1 0
    -/
  · exact one_ne_zero
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    h✝ : Not (∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.t …
    ⊢ Ne ⋯.choose 0
  -/
  exact (DilationClass.edist_eq' f).choose_spec.1
  /-
    🎉 no goals
  -/


theorem ratio_pos [DilationClass F α β] (f : F) : 0 < ratio f :=
  (ratio_ne_zero f).bot_lt


@[simp]
theorem edist_eq [DilationClass F α β] (f : F) (x y : α) :
    edist (f x) (f y) = ratio f * edist x y := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    x y : α
    ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑(Dilation.ratio f)) (EDist.edist x …
  -/
  rw [ratio]; split_ifs with key
    /-
      case pos
      α : Type u_1
      β : Type u_2
      F : Type u_4
      inst✝³ : PseudoEMetricSpace α
      inst✝² : PseudoEMetricSpace β
      inst✝¹ : FunLike F α β
      inst✝ : DilationClass F α β
      f : F
      x y : α
      key : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑1) (EDist.edist x y))
    -/
  · rcases DilationClass.edist_eq' f with ⟨r, hne, hr⟩
    /-
      case pos.intro.intro
      α : Type u_1
      β : Type u_2
      F : Type u_4
      inst✝³ : PseudoEMetricSpace α
      inst✝² : PseudoEMetricSpace β
      inst✝¹ : FunLike F α β
      inst✝ : DilationClass F α β
      f : F
      x y : α
      key : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      r : NNReal
      hne : Ne r 0
      hr : ∀ (x y : α), Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x  …
      ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑1) (EDist.edist x y))
    -/
    replace hr := hr x y
    /-
      case pos.intro.intro
      α : Type u_1
      β : Type u_2
      F : Type u_4
      inst✝³ : PseudoEMetricSpace α
      inst✝² : PseudoEMetricSpace β
      inst✝¹ : FunLike F α β
      inst✝ : DilationClass F α β
      f : F
      x y : α
      key : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      r : NNReal
      hne : Ne r 0
      hr : Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x y))
      ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑1) (EDist.edist x y))
    -/
    cases' key x y with h h
      /-
        case pos.intro.intro.inl
        α : Type u_1
        β : Type u_2
        F : Type u_4
        inst✝³ : PseudoEMetricSpace α
        inst✝² : PseudoEMetricSpace β
        inst✝¹ : FunLike F α β
        inst✝ : DilationClass F α β
        f : F
        x y : α
        key : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
        r : NNReal
        hne : Ne r 0
        hr : Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x y))
        h : Eq (EDist.edist x y) 0
        ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑1) (EDist.edist x y))
      -/
    · simp only [hr, h, mul_zero]
      /-
        🎉 no goals
      -/
      /-
        case pos.intro.intro.inr
        α : Type u_1
        β : Type u_2
        F : Type u_4
        inst✝³ : PseudoEMetricSpace α
        inst✝² : PseudoEMetricSpace β
        inst✝¹ : FunLike F α β
        inst✝ : DilationClass F α β
        f : F
        x y : α
        key : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
        r : NNReal
        hne : Ne r 0
        hr : Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x y))
        h : Eq (EDist.edist x y) Top.top
        ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑1) (EDist.edist x y))
      -/
    · simp [hr, h, hne]
      /-
        🎉 no goals
      -/
  /-
    case neg
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    x y : α
    key : Not (∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top. …
    ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑⋯.choose) (EDist.edist x y))
  -/
  exact (DilationClass.edist_eq' f).choose_spec.2 x y
  /-
    🎉 no goals
  -/


@[simp]
theorem nndist_eq {α β F : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β] [FunLike F α β]
    [DilationClass F α β] (f : F) (x y : α) :
    nndist (f x) (f y) = ratio f * nndist x y := by
  /-
    α : Type u_5
    β : Type u_6
    F : Type u_7
    inst✝³ : PseudoMetricSpace α
    inst✝² : PseudoMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    x y : α
    ⊢ Eq (NNDist.nndist (f x) (f y)) (HMul.hMul (Dilation.ratio f) (NNDist.nndist  …
  -/
  simp only [← ENNReal.coe_inj, ← edist_nndist, ENNReal.coe_mul, edist_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem dist_eq {α β F : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β] [FunLike F α β]
    [DilationClass F α β] (f : F) (x y : α) :
    dist (f x) (f y) = ratio f * dist x y := by
  /-
    α : Type u_5
    β : Type u_6
    F : Type u_7
    inst✝³ : PseudoMetricSpace α
    inst✝² : PseudoMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    x y : α
    ⊢ Eq (Dist.dist (f x) (f y)) (HMul.hMul (↑(Dilation.ratio f)) (Dist.dist x y))
  -/
  simp only [dist_nndist, nndist_eq, NNReal.coe_mul]
  /-
    🎉 no goals
  -/


/-- The `ratio` is equal to the distance ratio for any two points with nonzero finite distance.
`dist` and `nndist` versions below -/
theorem ratio_unique [DilationClass F α β] {f : F} {x y : α} {r : ℝ≥0} (h₀ : edist x y ≠ 0)
    (htop : edist x y ≠ ⊤) (hr : edist (f x) (f y) = r * edist x y) : r = ratio f := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    x y : α
    r : NNReal
    h₀ : Ne (EDist.edist x y) 0
    htop : Ne (EDist.edist x y) Top.top
    hr : Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x y))
    ⊢ Eq r (Dilation.ratio f)
  -/
  simpa only [hr, ENNReal.mul_eq_mul_right h₀ htop, ENNReal.coe_inj] using edist_eq f x y
  /-
    🎉 no goals
  -/


/-- The `ratio` is equal to the distance ratio for any two points
with nonzero finite distance; `nndist` version -/
theorem ratio_unique_of_nndist_ne_zero {α β F : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β]
    [FunLike F α β] [DilationClass F α β] {f : F} {x y : α} {r : ℝ≥0} (hxy : nndist x y ≠ 0)
    (hr : nndist (f x) (f y) = r * nndist x y) : r = ratio f :=
                   /-
                     α : Type u_5
                     β : Type u_6
                     F : Type u_7
                     inst✝³ : PseudoMetricSpace α
                     inst✝² : PseudoMetricSpace β
                     inst✝¹ : FunLike F α β
                     inst✝ : DilationClass F α β
                     f : F
                     x y : α
                     r : NNReal
                     hxy : Ne (NNDist.nndist x y) 0
                     hr : Eq (NNDist.nndist (f x) (f y)) (HMul.hMul r (NNDist.nndist x y))
                     ⊢ Ne (EDist.edist x y) 0
                   -/
  ratio_unique (by rwa [edist_nndist, ENNReal.coe_ne_zero]) (edist_ne_top x y)
                   /-
                     🎉 no goals
                   -/
        /-
          α : Type u_5
          β : Type u_6
          F : Type u_7
          inst✝³ : PseudoMetricSpace α
          inst✝² : PseudoMetricSpace β
          inst✝¹ : FunLike F α β
          inst✝ : DilationClass F α β
          f : F
          x y : α
          r : NNReal
          hxy : Ne (NNDist.nndist x y) 0
          hr : Eq (NNDist.nndist (f x) (f y)) (HMul.hMul r (NNDist.nndist x y))
          ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x y))
        -/
    (by rw [edist_nndist, edist_nndist, hr, ENNReal.coe_mul])
        /-
          🎉 no goals
        -/


/-- The `ratio` is equal to the distance ratio for any two points
with nonzero finite distance; `dist` version -/
theorem ratio_unique_of_dist_ne_zero {α β} {F : Type*} [PseudoMetricSpace α] [PseudoMetricSpace β]
    [FunLike F α β] [DilationClass F α β] {f : F} {x y : α} {r : ℝ≥0} (hxy : dist x y ≠ 0)
    (hr : dist (f x) (f y) = r * dist x y) : r = ratio f :=
  ratio_unique_of_nndist_ne_zero (NNReal.coe_ne_zero.1 hxy) <|
                    /-
                      α : Type u_6
                      β : Type u_7
                      F : Type u_5
                      inst✝³ : PseudoMetricSpace α
                      inst✝² : PseudoMetricSpace β
                      inst✝¹ : FunLike F α β
                      inst✝ : DilationClass F α β
                      f : F
                      x y : α
                      r : NNReal
                      hxy : Ne (Dist.dist x y) 0
                      hr : Eq (Dist.dist (f x) (f y)) (HMul.hMul (↑r) (Dist.dist x y))
                      ⊢ Eq ↑(NNDist.nndist (f x) (f y)) ↑(HMul.hMul r (NNDist.nndist x y))
                    -/
    NNReal.eq <| by rw [coe_nndist, hr, NNReal.coe_mul, coe_nndist]
                    /-
                      🎉 no goals
                    -/


/-- Alternative `Dilation` constructor when the distance hypothesis is over `nndist` -/
def mkOfNNDistEq {α β} [PseudoMetricSpace α] [PseudoMetricSpace β] (f : α → β)
    (h : ∃ r : ℝ≥0, r ≠ 0 ∧ ∀ x y : α, nndist (f x) (f y) = r * nndist x y) : α →ᵈ β where
  toFun := f
  edist_eq' := by
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ : Type u_3
      F : Type u_4
      inst✝⁴ : PseudoEMetricSpace α✝
      inst✝³ : PseudoEMetricSpace β✝
      inst✝² : FunLike F α✝ β✝
      α : Type ?u.18728
      β : Type ?u.18731
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      f : α → β
      h : Exists fun r => And (Ne r 0) (∀ (x y : α), Eq (NNDist.nndist (f x) (f y))  …
      ⊢ Exists fun r => And (Ne r 0) (∀ (x y : α), Eq (EDist.edist (f x) (f y)) (HMu …
    -/
    rcases h with ⟨r, hne, h⟩
    /-
      case intro.intro
      α✝ : Type u_1
      β✝ : Type u_2
      γ : Type u_3
      F : Type u_4
      inst✝⁴ : PseudoEMetricSpace α✝
      inst✝³ : PseudoEMetricSpace β✝
      inst✝² : FunLike F α✝ β✝
      α : Type ?u.18728
      β : Type ?u.18731
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      f : α → β
      r : NNReal
      hne : Ne r 0
      h : ∀ (x y : α), Eq (NNDist.nndist (f x) (f y)) (HMul.hMul r (NNDist.nndist x  …
      ⊢ Exists fun r => And (Ne r 0) (∀ (x y : α), Eq (EDist.edist (f x) (f y)) (HMu …
    -/
    refine ⟨r, hne, fun x y => ?_⟩
    /-
      case intro.intro
      α✝ : Type u_1
      β✝ : Type u_2
      γ : Type u_3
      F : Type u_4
      inst✝⁴ : PseudoEMetricSpace α✝
      inst✝³ : PseudoEMetricSpace β✝
      inst✝² : FunLike F α✝ β✝
      α : Type ?u.18728
      β : Type ?u.18731
      inst✝¹ : PseudoMetricSpace α
      inst✝ : PseudoMetricSpace β
      f : α → β
      r : NNReal
      hne : Ne r 0
      h : ∀ (x y : α), Eq (NNDist.nndist (f x) (f y)) (HMul.hMul r (NNDist.nndist x  …
      x y : α
      ⊢ Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑r) (EDist.edist x y))
    -/
    rw [edist_nndist, edist_nndist, ← ENNReal.coe_mul, h x y]
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_mkOfNNDistEq {α β} [PseudoMetricSpace α] [PseudoMetricSpace β] (f : α → β) (h) :
    ⇑(mkOfNNDistEq f h : α →ᵈ β) = f :=
  rfl


@[simp]
theorem mk_coe_of_nndist_eq {α β} [PseudoMetricSpace α] [PseudoMetricSpace β] (f : α →ᵈ β)
    (h) : Dilation.mkOfNNDistEq f h = f :=
  ext fun _ => rfl


/-- Alternative `Dilation` constructor when the distance hypothesis is over `dist` -/
def mkOfDistEq {α β} [PseudoMetricSpace α] [PseudoMetricSpace β] (f : α → β)
    (h : ∃ r : ℝ≥0, r ≠ 0 ∧ ∀ x y : α, dist (f x) (f y) = r * dist x y) : α →ᵈ β :=
  mkOfNNDistEq f <|
    h.imp fun r hr =>
                                        /-
                                          α✝ : Type u_1
                                          β✝ : Type u_2
                                          γ : Type u_3
                                          F : Type u_4
                                          inst✝⁴ : PseudoEMetricSpace α✝
                                          inst✝³ : PseudoEMetricSpace β✝
                                          inst✝² : FunLike F α✝ β✝
                                          α : Type ?u.20022
                                          β : Type ?u.20025
                                          inst✝¹ : PseudoMetricSpace α
                                          inst✝ : PseudoMetricSpace β
                                          f : α → β
                                          h : Exists fun r => And (Ne r 0) (∀ (x y : α), Eq (Dist.dist (f x) (f y)) (HMu …
                                          r : NNReal
                                          hr : And (Ne r 0) (∀ (x y : α), Eq (Dist.dist (f x) (f y)) (HMul.hMul (↑r) (Di …
                                          x y : α
                                          ⊢ Eq ↑(NNDist.nndist (f x) (f y)) ↑(HMul.hMul r (NNDist.nndist x y))
                                        -/
      ⟨hr.1, fun x y => NNReal.eq <| by rw [coe_nndist, hr.2, NNReal.coe_mul, coe_nndist]⟩
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem coe_mkOfDistEq {α β} [PseudoMetricSpace α] [PseudoMetricSpace β] (f : α → β) (h) :
    ⇑(mkOfDistEq f h : α →ᵈ β) = f :=
  rfl


@[simp]
theorem mk_coe_of_dist_eq {α β} [PseudoMetricSpace α] [PseudoMetricSpace β] (f : α →ᵈ β) (h) :
    Dilation.mkOfDistEq f h = f :=
  ext fun _ => rfl


/-- Every isometry is a dilation of ratio `1`. -/
@[simps]
def _root_.Isometry.toDilation (f : α → β) (hf : Isometry f) : α →ᵈ β where
  toFun := f
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     γ : Type u_3
                                     F : Type u_4
                                     inst✝⁴ : PseudoEMetricSpace α
                                     inst✝³ : PseudoEMetricSpace β
                                     inst✝² : PseudoEMetricSpace γ
                                     inst✝¹ : FunLike F α β
                                     inst✝ : DilationClass F α β
                                     f✝ : F
                                     f : α → β
                                     hf : Isometry f
                                     ⊢ ∀ (x y : α), Eq (EDist.edist (f x) (f y)) (HMul.hMul (↑1) (EDist.edist x y))
                                   -/
  edist_eq' := ⟨1, one_ne_zero, by simpa using hf⟩
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
lemma _root_.Isometry.toDilation_ratio {f : α → β} {hf : Isometry f} : ratio hf.toDilation = 1 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PseudoEMetricSpace α
    inst✝ : PseudoEMetricSpace β
    f : α → β
    hf : Isometry f
    ⊢ Eq (Dilation.ratio (Isometry.toDilation f hf)) 1
  -/
  by_cases h : ∀ x y : α, edist x y = 0 ∨ edist x y = ⊤
    /-
      case pos
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      f : α → β
      hf : Isometry f
      h : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      ⊢ Eq (Dilation.ratio (Isometry.toDilation f hf)) 1
    -/
  · exact ratio_of_trivial hf.toDilation h
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      f : α → β
      hf : Isometry f
      h : Not (∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.to …
      ⊢ Eq (Dilation.ratio (Isometry.toDilation f hf)) 1
    -/
  · push_neg at h
    /-
      case neg
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      f : α → β
      hf : Isometry f
      h : Exists fun x => Exists fun y => And (Ne (EDist.edist x y) 0) (Ne (EDist.ed …
      ⊢ Eq (Dilation.ratio (Isometry.toDilation f hf)) 1
    -/
    obtain ⟨x, y, h₁, h₂⟩ := h
    /-
      case neg.intro.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝¹ : PseudoEMetricSpace α
      inst✝ : PseudoEMetricSpace β
      f : α → β
      hf : Isometry f
      x y : α
      h₁ : Ne (EDist.edist x y) 0
      h₂ : Ne (EDist.edist x y) Top.top
      ⊢ Eq (Dilation.ratio (Isometry.toDilation f hf)) 1
    -/
    exact ratio_unique h₁ h₂ (by simp [hf x y]) |>.symm
    /-
      🎉 no goals
    -/


theorem lipschitz : LipschitzWith (ratio f) (f : α → β) := fun x y => (edist_eq f x y).le


theorem antilipschitz : AntilipschitzWith (ratio f)⁻¹ (f : α → β) := fun x y => by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    x y : α
    ⊢ LE.le (EDist.edist x y) (HMul.hMul (↑(Inv.inv (Dilation.ratio f))) (EDist.ed …
  -/
  have hr : ratio f ≠ 0 := ratio_ne_zero f
  exact mod_cast
    (ENNReal.mul_le_iff_le_inv (ENNReal.coe_ne_zero.2 hr) ENNReal.coe_ne_top).1 (edist_eq f x y).ge


/-- A dilation from an emetric space is injective -/
protected theorem injective {α : Type*} [EMetricSpace α] [FunLike F α β]  [DilationClass F α β]
    (f : F) :
    Injective f :=
  (antilipschitz f).injective


/-- The identity is a dilation -/
protected def id (α) [PseudoEMetricSpace α] : α →ᵈ α where
  toFun := id
                                              /-
                                                α✝ : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                F : Type u_4
                                                inst✝⁵ : PseudoEMetricSpace α✝
                                                inst✝⁴ : PseudoEMetricSpace β
                                                inst✝³ : PseudoEMetricSpace γ
                                                inst✝² : FunLike F α✝ β
                                                inst✝¹ : DilationClass F α✝ β
                                                f : F
                                                α : Type ?u.29485
                                                inst✝ : PseudoEMetricSpace α
                                                x y : α
                                                ⊢ Eq (EDist.edist (id x) (id y)) (HMul.hMul (↑1) (EDist.edist x y))
                                              -/
  edist_eq' := ⟨1, one_ne_zero, fun x y => by simp only [id, ENNReal.coe_one, one_mul]⟩
                                              /-
                                                🎉 no goals
                                              -/


instance : Inhabited (α →ᵈ α) :=
  ⟨Dilation.id α⟩


@[simp]
protected theorem coe_id : ⇑(Dilation.id α) = id :=
  rfl


theorem ratio_id : ratio (Dilation.id α) = 1 := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    ⊢ Eq (Dilation.ratio (Dilation.id α)) 1
  -/
  by_cases h : ∀ x y : α, edist x y = 0 ∨ edist x y = ∞
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      h : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      ⊢ Eq (Dilation.ratio (Dilation.id α)) 1
    -/
  · rw [ratio, if_pos h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      h : Not (∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.to …
      ⊢ Eq (Dilation.ratio (Dilation.id α)) 1
    -/
  · push_neg at h
    /-
      case neg
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      h : Exists fun x => Exists fun y => And (Ne (EDist.edist x y) 0) (Ne (EDist.ed …
      ⊢ Eq (Dilation.ratio (Dilation.id α)) 1
    -/
    rcases h with ⟨x, y, hne⟩
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      x y : α
      hne : And (Ne (EDist.edist x y) 0) (Ne (EDist.edist x y) Top.top)
      ⊢ Eq (Dilation.ratio (Dilation.id α)) 1
    -/
    refine (ratio_unique hne.1 hne.2 ?_).symm
    /-
      case neg.intro.intro
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      x y : α
      hne : And (Ne (EDist.edist x y) 0) (Ne (EDist.edist x y) Top.top)
      ⊢ Eq (EDist.edist ((Dilation.id α) x) ((Dilation.id α) y)) (HMul.hMul (↑1) (ED …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- The composition of dilations is a dilation -/
def comp (g : β →ᵈ γ) (f : α →ᵈ β) : α →ᵈ γ where
  toFun := g ∘ f
  edist_eq' := ⟨ratio g * ratio f, mul_ne_zero (ratio_ne_zero g) (ratio_ne_zero f),
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    F : Type u_4
                    inst✝⁴ : PseudoEMetricSpace α
                    inst✝³ : PseudoEMetricSpace β
                    inst✝² : PseudoEMetricSpace γ
                    inst✝¹ : FunLike F α β
                    inst✝ : DilationClass F α β
                    f✝ : F
                    g : Dilation β γ
                    f : Dilation α β
                    x y : α
                    ⊢ Eq (EDist.edist (Function.comp (⇑g) (⇑f) x) (Function.comp (⇑g) (⇑f) y)) (HM …
                  -/
    fun x y => by simp_rw [Function.comp, edist_eq, ENNReal.coe_mul, mul_assoc]⟩
                  /-
                    🎉 no goals
                  -/


theorem comp_assoc {δ : Type*} [PseudoEMetricSpace δ] (f : α →ᵈ β) (g : β →ᵈ γ)
    (h : γ →ᵈ δ) : (h.comp g).comp f = h.comp (g.comp f) :=
  rfl


@[simp]
theorem coe_comp (g : β →ᵈ γ) (f : α →ᵈ β) : (g.comp f : α → γ) = g ∘ f :=
  rfl


theorem comp_apply (g : β →ᵈ γ) (f : α →ᵈ β) (x : α) : (g.comp f : α → γ) x = g (f x) :=
  rfl

-- Porting note: removed `simp` because it's difficult to auto prove `hne`

/-- Ratio of the composition `g.comp f` of two dilations is the product of their ratios. We assume
that there exist two points in `α` at extended distance neither `0` nor `∞` because otherwise
`Dilation.ratio (g.comp f) = Dilation.ratio f = 1` while `Dilation.ratio g` can be any number. This
version works for most general spaces, see also `Dilation.ratio_comp` for a version assuming that
`α` is a nontrivial metric space. -/
theorem ratio_comp' {g : β →ᵈ γ} {f : α →ᵈ β}
    (hne : ∃ x y : α, edist x y ≠ 0 ∧ edist x y ≠ ⊤) : ratio (g.comp f) = ratio g * ratio f := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : PseudoEMetricSpace β
    inst✝ : PseudoEMetricSpace γ
    g : Dilation β γ
    f : Dilation α β
    hne : Exists fun x => Exists fun y => And (Ne (EDist.edist x y) 0) (Ne (EDist. …
    ⊢ Eq (Dilation.ratio (g.comp f)) (HMul.hMul (Dilation.ratio g) (Dilation.ratio …
  -/
  rcases hne with ⟨x, y, hα⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : PseudoEMetricSpace β
    inst✝ : PseudoEMetricSpace γ
    g : Dilation β γ
    f : Dilation α β
    x y : α
    hα : And (Ne (EDist.edist x y) 0) (Ne (EDist.edist x y) Top.top)
    ⊢ Eq (Dilation.ratio (g.comp f)) (HMul.hMul (Dilation.ratio g) (Dilation.ratio …
  -/
  have hgf := (edist_eq (g.comp f) x y).symm
  simp_rw [coe_comp, Function.comp, edist_eq, ← mul_assoc, ENNReal.mul_eq_mul_right hα.1 hα.2]
    at hgf
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝² : PseudoEMetricSpace α
    inst✝¹ : PseudoEMetricSpace β
    inst✝ : PseudoEMetricSpace γ
    g : Dilation β γ
    f : Dilation α β
    x y : α
    hα : And (Ne (EDist.edist x y) 0) (Ne (EDist.edist x y) Top.top)
    hgf : Eq (↑(Dilation.ratio (g.comp f))) (HMul.hMul ↑(Dilation.ratio g) ↑(Dilat …
    ⊢ Eq (Dilation.ratio (g.comp f)) (HMul.hMul (Dilation.ratio g) (Dilation.ratio …
  -/
  rwa [← ENNReal.coe_inj, ENNReal.coe_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_id (f : α →ᵈ β) : f.comp (Dilation.id α) = f :=
  ext fun _ => rfl


@[simp]
theorem id_comp (f : α →ᵈ β) : (Dilation.id β).comp f = f :=
  ext fun _ => rfl


instance : Monoid (α →ᵈ α) where
  one := Dilation.id α
  mul := comp
  mul_one := comp_id
  one_mul := id_comp
  mul_assoc _ _ _ := comp_assoc _ _ _


theorem one_def : (1 : α →ᵈ α) = Dilation.id α :=
  rfl


theorem mul_def (f g : α →ᵈ α) : f * g = f.comp g :=
  rfl


@[simp]
theorem coe_one : ⇑(1 : α →ᵈ α) = id :=
  rfl


@[simp]
theorem coe_mul (f g : α →ᵈ α) : ⇑(f * g) = f ∘ g :=
  rfl


@[simp] theorem ratio_one : ratio (1 : α →ᵈ α) = 1 := ratio_id


@[simp]
theorem ratio_mul (f g : α →ᵈ α) : ratio (f * g) = ratio f * ratio g := by
  /-
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f g : Dilation α α
    ⊢ Eq (Dilation.ratio (HMul.hMul f g)) (HMul.hMul (Dilation.ratio f) (Dilation. …
  -/
  by_cases h : ∀ x y : α, edist x y = 0 ∨ edist x y = ∞
    /-
      case pos
      α : Type u_1
      inst✝ : PseudoEMetricSpace α
      f g : Dilation α α
      h : ∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.top)
      ⊢ Eq (Dilation.ratio (HMul.hMul f g)) (HMul.hMul (Dilation.ratio f) (Dilation. …
    -/
  · simp [ratio_of_trivial, h]
    /-
      🎉 no goals
    -/
  /-
    case neg
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f g : Dilation α α
    h : Not (∀ (x y : α), Or (Eq (EDist.edist x y) 0) (Eq (EDist.edist x y) Top.to …
    ⊢ Eq (Dilation.ratio (HMul.hMul f g)) (HMul.hMul (Dilation.ratio f) (Dilation. …
  -/
  push_neg at h
  /-
    case neg
    α : Type u_1
    inst✝ : PseudoEMetricSpace α
    f g : Dilation α α
    h : Exists fun x => Exists fun y => And (Ne (EDist.edist x y) 0) (Ne (EDist.ed …
    ⊢ Eq (Dilation.ratio (HMul.hMul f g)) (HMul.hMul (Dilation.ratio f) (Dilation. …
  -/
  exact ratio_comp' h
  /-
    🎉 no goals
  -/


/-- `Dilation.ratio` as a monoid homomorphism from `α →ᵈ α` to `ℝ≥0`. -/
@[simps]
def ratioHom : (α →ᵈ α) →* ℝ≥0 := ⟨⟨ratio, ratio_one⟩, ratio_mul⟩


@[simp]
theorem ratio_pow (f : α →ᵈ α) (n : ℕ) : ratio (f ^ n) = ratio f ^ n :=
  ratioHom.map_pow _ _


@[simp]
theorem cancel_right {g₁ g₂ : β →ᵈ γ} {f : α →ᵈ β} (hf : Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => Dilation.ext <| hf.forall.2 (Dilation.ext_iff.1 h), fun h => h ▸ rfl⟩


@[simp]
theorem cancel_left {g : β →ᵈ γ} {f₁ f₂ : α →ᵈ β} (hg : Injective g) :
    g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             γ : Type u_3
                                             inst✝² : PseudoEMetricSpace α
                                             inst✝¹ : PseudoEMetricSpace β
                                             inst✝ : PseudoEMetricSpace γ
                                             g : Dilation β γ
                                             f₁ f₂ : Dilation α β
                                             hg : Function.Injective ⇑g
                                             h : Eq (g.comp f₁) (g.comp f₂)
                                             x : α
                                             ⊢ Eq (g (f₁ x)) (g (f₂ x))
                                           -/
  ⟨fun h => Dilation.ext fun x => hg <| by rw [← comp_apply, h, comp_apply], fun h => h ▸ rfl⟩
                                           /-
                                             🎉 no goals
                                           -/


/-- A dilation from a metric space is a uniform inducing map -/
theorem isUniformInducing : IsUniformInducing (f : α → β) :=
  (antilipschitz f).isUniformInducing (lipschitz f).uniformContinuous


@[deprecated (since := "2024-10-05")]
alias uniformInducing := isUniformInducing


theorem tendsto_nhds_iff {ι : Type*} {g : ι → α} {a : Filter ι} {b : α} :
    Filter.Tendsto g a (𝓝 b) ↔ Filter.Tendsto ((f : α → β) ∘ g) a (𝓝 (f b)) :=
  (Dilation.isUniformInducing f).isInducing.tendsto_nhds_iff


/-- A dilation is continuous. -/
theorem toContinuous : Continuous (f : α → β) :=
  (lipschitz f).continuous


/-- Dilations scale the diameter by `ratio f` in pseudoemetric spaces. -/
theorem ediam_image (s : Set α) : EMetric.diam ((f : α → β) '' s) = ratio f * EMetric.diam s := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    s : Set α
    ⊢ Eq (EMetric.diam (Set.image (⇑f) s)) (HMul.hMul (↑(Dilation.ratio f)) (EMetr …
  -/
  refine ((lipschitz f).ediam_image_le s).antisymm ?_
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    s : Set α
    ⊢ LE.le (HMul.hMul (↑(Dilation.ratio f)) (EMetric.diam s)) (EMetric.diam (Set. …
  -/
  apply ENNReal.mul_le_of_le_div'
  /-
    case h
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    s : Set α
    ⊢ LE.le (EMetric.diam s) (HDiv.hDiv (EMetric.diam (Set.image (⇑f) s)) ↑(Dilati …
  -/
  rw [div_eq_mul_inv, mul_comm, ← ENNReal.coe_inv]
  /-
    case h
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    s : Set α
    ⊢ LE.le (EMetric.diam s) (HMul.hMul (↑(Inv.inv (Dilation.ratio f))) (EMetric.d …
  -/
  exacts [(antilipschitz f).le_mul_ediam_image s, ratio_ne_zero f]
  /-
    🎉 no goals
  -/


/-- A dilation scales the diameter of the range by `ratio f`. -/
theorem ediam_range : EMetric.diam (range (f : α → β)) = ratio f * EMetric.diam (univ : Set α) := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoEMetricSpace α
    inst✝² : PseudoEMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    ⊢ Eq (EMetric.diam (Set.range ⇑f)) (HMul.hMul (↑(Dilation.ratio f)) (EMetric.d …
  -/
  rw [← image_univ]; exact ediam_image f univ
                     /-
                       🎉 no goals
                     -/


/-- A dilation maps balls to balls and scales the radius by `ratio f`. -/
theorem mapsTo_emetric_ball (x : α) (r : ℝ≥0∞) :
    MapsTo (f : α → β) (EMetric.ball x r) (EMetric.ball (f x) (ratio f * r)) :=
  fun y hy => (edist_eq f y x).trans_lt <|
    (ENNReal.mul_lt_mul_left (ENNReal.coe_ne_zero.2 <| ratio_ne_zero f) ENNReal.coe_ne_top).2 hy


/-- A dilation maps closed balls to closed balls and scales the radius by `ratio f`. -/
theorem mapsTo_emetric_closedBall (x : α) (r' : ℝ≥0∞) :
    MapsTo (f : α → β) (EMetric.closedBall x r') (EMetric.closedBall (f x) (ratio f * r')) :=
  -- Porting note: Added `by exact`
                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  F : Type u_4
                                                                  inst✝³ : PseudoEMetricSpace α
                                                                  inst✝² : PseudoEMetricSpace β
                                                                  inst✝¹ : FunLike F α β
                                                                  inst✝ : DilationClass F α β
                                                                  f : F
                                                                  x : α
                                                                  r' : ENNReal
                                                                  y : α
                                                                  hy : Membership.mem (EMetric.closedBall x r') y
                                                                  ⊢ LE.le (EDist.edist y x) r'
                                                                -/
  fun y hy => (edist_eq f y x).trans_le <| mul_le_mul_left' (by exact hy) _
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem comp_continuousOn_iff {γ} [TopologicalSpace γ] {g : γ → α} {s : Set γ} :
    ContinuousOn ((f : α → β) ∘ g) s ↔ ContinuousOn g s :=
  (Dilation.isUniformInducing f).isInducing.continuousOn_iff.symm


theorem comp_continuous_iff {γ} [TopologicalSpace γ] {g : γ → α} :
    Continuous ((f : α → β) ∘ g) ↔ Continuous g :=
  (Dilation.isUniformInducing f).isInducing.continuous_iff.symm


/-- A dilation from a metric space is a uniform embedding -/
lemma isUniformEmbedding [PseudoEMetricSpace β] [DilationClass F α β] (f : F) :
    IsUniformEmbedding f :=
  (antilipschitz f).isUniformEmbedding (lipschitz f).uniformContinuous


@[deprecated (since := "2024-10-01")] alias uniformEmbedding := isUniformEmbedding


/-- A dilation from a metric space is an embedding -/
theorem isEmbedding [PseudoEMetricSpace β] [DilationClass F α β] (f : F) :
    IsEmbedding (f : α → β) :=
  (Dilation.isUniformEmbedding f).isEmbedding


@[deprecated (since := "2024-10-26")]
alias embedding := isEmbedding


/-- A dilation from a complete emetric space is a closed embedding -/
lemma isClosedEmbedding [CompleteSpace α] [EMetricSpace β] [DilationClass F α β] (f : F) :
    IsClosedEmbedding f :=
  (antilipschitz f).isClosedEmbedding (lipschitz f).uniformContinuous


@[deprecated (since := "2024-10-20")] alias closedEmbedding := isClosedEmbedding


/-- Ratio of the composition `g.comp f` of two dilations is the product of their ratios. We assume
that the domain `α` of `f` is a nontrivial metric space, otherwise
`Dilation.ratio f = Dilation.ratio (g.comp f) = 1` but `Dilation.ratio g` may have any value.

See also `Dilation.ratio_comp'` for a version that works for more general spaces. -/
@[simp]
theorem ratio_comp [MetricSpace α] [Nontrivial α] [PseudoEMetricSpace β]
    [PseudoEMetricSpace γ] {g : β →ᵈ γ} {f : α →ᵈ β} : ratio (g.comp f) = ratio g * ratio f :=
  ratio_comp' <|
    let ⟨x, y, hne⟩ := exists_pair_ne α; ⟨x, y, mt edist_eq_zero.1 hne, edist_ne_top _ _⟩


/-- A dilation scales the diameter by `ratio f` in pseudometric spaces. -/
theorem diam_image (s : Set α) : Metric.diam ((f : α → β) '' s) = ratio f * Metric.diam s := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoMetricSpace α
    inst✝² : PseudoMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    s : Set α
    ⊢ Eq (Metric.diam (Set.image (⇑f) s)) (HMul.hMul (↑(Dilation.ratio f)) (Metric …
  -/
  simp [Metric.diam, ediam_image, ENNReal.toReal_mul]
  /-
    🎉 no goals
  -/


theorem diam_range : Metric.diam (range (f : α → β)) = ratio f * Metric.diam (univ : Set α) := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_4
    inst✝³ : PseudoMetricSpace α
    inst✝² : PseudoMetricSpace β
    inst✝¹ : FunLike F α β
    inst✝ : DilationClass F α β
    f : F
    ⊢ Eq (Metric.diam (Set.range ⇑f)) (HMul.hMul (↑(Dilation.ratio f)) (Metric.dia …
  -/
  rw [← image_univ, diam_image]
  /-
    🎉 no goals
  -/


/-- A dilation maps balls to balls and scales the radius by `ratio f`. -/
theorem mapsTo_ball (x : α) (r' : ℝ) :
    MapsTo (f : α → β) (Metric.ball x r') (Metric.ball (f x) (ratio f * r')) :=
  fun y hy => (dist_eq f y x).trans_lt <| (mul_lt_mul_left <| NNReal.coe_pos.2 <| ratio_pos f).2 hy


/-- A dilation maps spheres to spheres and scales the radius by `ratio f`. -/
theorem mapsTo_sphere (x : α) (r' : ℝ) :
    MapsTo (f : α → β) (Metric.sphere x r') (Metric.sphere (f x) (ratio f * r')) :=
  fun y hy => Metric.mem_sphere.mp hy ▸ dist_eq f y x


/-- A dilation maps closed balls to closed balls and scales the radius by `ratio f`. -/
theorem mapsTo_closedBall (x : α) (r' : ℝ) :
    MapsTo (f : α → β) (Metric.closedBall x r') (Metric.closedBall (f x) (ratio f * r')) :=
  fun y hy => (dist_eq f y x).trans_le <| mul_le_mul_of_nonneg_left hy (NNReal.coe_nonneg _)


lemma tendsto_cobounded : Filter.Tendsto f (cobounded α) (cobounded β) :=
  (Dilation.antilipschitz f).tendsto_cobounded


@[simp]
lemma comap_cobounded : Filter.comap f (cobounded β) = cobounded α :=
  le_antisymm (lipschitz f).comap_cobounded_le (tendsto_cobounded f).le_comap


