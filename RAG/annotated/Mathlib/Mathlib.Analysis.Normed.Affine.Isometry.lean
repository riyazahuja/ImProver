/-- A `𝕜`-affine isometric embedding of one normed add-torsor over a normed `𝕜`-space into
another. -/
structure AffineIsometry extends P →ᵃ[𝕜] P₂ where
  norm_map : ∀ x : V, ‖linear x‖ = ‖x‖


@[inherit_doc]
notation:25 -- `→ᵃᵢ` would be more consistent with the linear isometry notation, but it is uglier
P " →ᵃⁱ[" 𝕜:25 "] " P₂:0 => AffineIsometry 𝕜 P P₂


/-- The underlying linear map of an affine isometry is in fact a linear isometry. -/
protected def linearIsometry : V →ₗᵢ[𝕜] V₂ :=
  { f.linear with norm_map' := f.norm_map }


@[simp]
theorem linear_eq_linearIsometry : f.linear = f.linearIsometry.toLinearMap := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineIsometry 𝕜 P P₂
    ⊢ Eq f.linear f.linearIsometry.toLinearMap
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineIsometry 𝕜 P P₂
    x✝ : V
    ⊢ Eq (f.linear x✝) (f.linearIsometry.toLinearMap x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : FunLike (P →ᵃⁱ[𝕜] P₂) P P₂ where
  coe f := f.toFun
                           /-
                             𝕜 : Type u_1
                             V : Type u_2
                             V₁ : Type u_3
                             V₁' : Type u_4
                             V₂ : Type u_5
                             V₃ : Type u_6
                             V₄ : Type u_7
                             P₁ : Type u_8
                             P₁' : Type u_9
                             P : Type u_10
                             P₂ : Type u_11
                             P₃ : Type u_12
                             P₄ : Type u_13
                             inst✝²⁴ : NormedField 𝕜
                             inst✝²³ : SeminormedAddCommGroup V
                             inst✝²² : NormedSpace 𝕜 V
                             inst✝²¹ : PseudoMetricSpace P
                             inst✝²⁰ : NormedAddTorsor V P
                             inst✝¹⁹ : SeminormedAddCommGroup V₁
                             inst✝¹⁸ : NormedSpace 𝕜 V₁
                             inst✝¹⁷ : PseudoMetricSpace P₁
                             inst✝¹⁶ : NormedAddTorsor V₁ P₁
                             inst✝¹⁵ : SeminormedAddCommGroup V₁'
                             inst✝¹⁴ : NormedSpace 𝕜 V₁'
                             inst✝¹³ : MetricSpace P₁'
                             inst✝¹² : NormedAddTorsor V₁' P₁'
                             inst✝¹¹ : SeminormedAddCommGroup V₂
                             inst✝¹⁰ : NormedSpace 𝕜 V₂
                             inst✝⁹ : PseudoMetricSpace P₂
                             inst✝⁸ : NormedAddTorsor V₂ P₂
                             inst✝⁷ : SeminormedAddCommGroup V₃
                             inst✝⁶ : NormedSpace 𝕜 V₃
                             inst✝⁵ : PseudoMetricSpace P₃
                             inst✝⁴ : NormedAddTorsor V₃ P₃
                             inst✝³ : SeminormedAddCommGroup V₄
                             inst✝² : NormedSpace 𝕜 V₄
                             inst✝¹ : PseudoMetricSpace P₄
                             inst✝ : NormedAddTorsor V₄ P₄
                             f✝ f g : AffineIsometry 𝕜 P P₂
                             ⊢ Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g) → Eq f g
                           -/
  coe_injective' f g := by cases f; cases g; simp
                                             /-
                                               🎉 no goals
                                             -/


@[simp]
theorem coe_toAffineMap : ⇑f.toAffineMap = f := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineIsometry 𝕜 P P₂
    ⊢ Eq ⇑f.toAffineMap ⇑f
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem toAffineMap_injective : Injective (toAffineMap : (P →ᵃⁱ[𝕜] P₂) → P →ᵃ[𝕜] P₂) := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    ⊢ Function.Injective AffineIsometry.toAffineMap
  -/
  rintro ⟨f, _⟩ ⟨g, _⟩ rfl
  /-
    case mk.mk
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    norm_map✝¹ : ∀ (x : V), Eq (Norm.norm (f.linear x)) (Norm.norm x)
    norm_map✝ : ∀ (x : V), Eq (Norm.norm ({ toAffineMap := f, norm_map := norm_map …
    ⊢ Eq { toAffineMap := f, norm_map := norm_map✝¹ } { toAffineMap := { toAffineM …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem coeFn_injective : @Injective (P →ᵃⁱ[𝕜] P₂) (P → P₂) (↑) :=
  AffineMap.coeFn_injective.comp toAffineMap_injective


@[ext]
theorem ext {f g : P →ᵃⁱ[𝕜] P₂} (h : ∀ x, f x = g x) : f = g :=
  coeFn_injective <| funext h


/-- Reinterpret a linear isometry as an affine isometry. -/
def toAffineIsometry : V →ᵃⁱ[𝕜] V₂ :=
  { f.toLinearMap.toAffineMap with norm_map := f.norm_map }


@[simp]
theorem coe_toAffineIsometry : ⇑(f.toAffineIsometry : V →ᵃⁱ[𝕜] V₂) = f :=
  rfl


@[simp]
theorem toAffineIsometry_linearIsometry : f.toAffineIsometry.linearIsometry = f := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : NormedSpace 𝕜 V₂
    f : LinearIsometry (RingHom.id 𝕜) V V₂
    ⊢ Eq f.toAffineIsometry.linearIsometry f
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : NormedSpace 𝕜 V₂
    f : LinearIsometry (RingHom.id 𝕜) V V₂
    x✝ : V
    ⊢ Eq (f.toAffineIsometry.linearIsometry x✝) (f x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- somewhat arbitrary choice of simp direction

@[simp]
theorem toAffineIsometry_toAffineMap : f.toAffineIsometry.toAffineMap = f.toLinearMap.toAffineMap :=
  rfl


@[simp]
theorem map_vadd (p : P) (v : V) : f (v +ᵥ p) = f.linearIsometry v +ᵥ f p :=
  f.toAffineMap.map_vadd p v


@[simp]
theorem map_vsub (p1 p2 : P) : f.linearIsometry (p1 -ᵥ p2) = f p1 -ᵥ f p2 :=
  f.toAffineMap.linearMap_vsub p1 p2


@[simp]
theorem dist_map (x y : P) : dist (f x) (f y) = dist x y := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineIsometry 𝕜 P P₂
    x y : P
    ⊢ Eq (Dist.dist (f x) (f y)) (Dist.dist x y)
  -/
  rw [dist_eq_norm_vsub V₂, dist_eq_norm_vsub V, ← map_vsub, f.linearIsometry.norm_map]
  /-
    🎉 no goals
  -/


@[simp]
                                                                     /-
                                                                       𝕜 : Type u_1
                                                                       V : Type u_2
                                                                       V₂ : Type u_5
                                                                       P : Type u_10
                                                                       P₂ : Type u_11
                                                                       inst✝⁸ : NormedField 𝕜
                                                                       inst✝⁷ : SeminormedAddCommGroup V
                                                                       inst✝⁶ : NormedSpace 𝕜 V
                                                                       inst✝⁵ : PseudoMetricSpace P
                                                                       inst✝⁴ : NormedAddTorsor V P
                                                                       inst✝³ : SeminormedAddCommGroup V₂
                                                                       inst✝² : NormedSpace 𝕜 V₂
                                                                       inst✝¹ : PseudoMetricSpace P₂
                                                                       inst✝ : NormedAddTorsor V₂ P₂
                                                                       f : AffineIsometry 𝕜 P P₂
                                                                       x y : P
                                                                       ⊢ Eq (NNDist.nndist (f x) (f y)) (NNDist.nndist x y)
                                                                     -/
theorem nndist_map (x y : P) : nndist (f x) (f y) = nndist x y := by simp [nndist_dist]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
                                                                  /-
                                                                    𝕜 : Type u_1
                                                                    V : Type u_2
                                                                    V₂ : Type u_5
                                                                    P : Type u_10
                                                                    P₂ : Type u_11
                                                                    inst✝⁸ : NormedField 𝕜
                                                                    inst✝⁷ : SeminormedAddCommGroup V
                                                                    inst✝⁶ : NormedSpace 𝕜 V
                                                                    inst✝⁵ : PseudoMetricSpace P
                                                                    inst✝⁴ : NormedAddTorsor V P
                                                                    inst✝³ : SeminormedAddCommGroup V₂
                                                                    inst✝² : NormedSpace 𝕜 V₂
                                                                    inst✝¹ : PseudoMetricSpace P₂
                                                                    inst✝ : NormedAddTorsor V₂ P₂
                                                                    f : AffineIsometry 𝕜 P P₂
                                                                    x y : P
                                                                    ⊢ Eq (EDist.edist (f x) (f y)) (EDist.edist x y)
                                                                  -/
theorem edist_map (x y : P) : edist (f x) (f y) = edist x y := by simp [edist_dist]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


protected theorem isometry : Isometry f :=
  f.edist_map


protected theorem injective : Injective f₁ :=
  f₁.isometry.injective


@[simp]
theorem map_eq_iff {x y : P₁'} : f₁ x = f₁ y ↔ x = y :=
  f₁.injective.eq_iff


theorem map_ne {x y : P₁'} (h : x ≠ y) : f₁ x ≠ f₁ y :=
  f₁.injective.ne h


protected theorem lipschitz : LipschitzWith 1 f :=
  f.isometry.lipschitz


protected theorem antilipschitz : AntilipschitzWith 1 f :=
  f.isometry.antilipschitz


@[continuity]
protected theorem continuous : Continuous f :=
  f.isometry.continuous


theorem ediam_image (s : Set P) : EMetric.diam (f '' s) = EMetric.diam s :=
  f.isometry.ediam_image s


theorem ediam_range : EMetric.diam (range f) = EMetric.diam (univ : Set P) :=
  f.isometry.ediam_range


theorem diam_image (s : Set P) : Metric.diam (f '' s) = Metric.diam s :=
  f.isometry.diam_image s


theorem diam_range : Metric.diam (range f) = Metric.diam (univ : Set P) :=
  f.isometry.diam_range


@[simp]
theorem comp_continuous_iff {α : Type*} [TopologicalSpace α] {g : α → P} :
    Continuous (f ∘ g) ↔ Continuous g :=
  f.isometry.comp_continuous_iff


/-- The identity affine isometry. -/
def id : P →ᵃⁱ[𝕜] P :=
  ⟨AffineMap.id 𝕜 P, fun _ => rfl⟩


@[simp]
theorem coe_id : ⇑(id : P →ᵃⁱ[𝕜] P) = _root_.id :=
  rfl


@[simp]
theorem id_apply (x : P) : (AffineIsometry.id : P →ᵃⁱ[𝕜] P) x = x :=
  rfl


@[simp]
theorem id_toAffineMap : (id.toAffineMap : P →ᵃ[𝕜] P) = AffineMap.id 𝕜 P :=
  rfl


instance : Inhabited (P →ᵃⁱ[𝕜] P) :=
  ⟨id⟩


/-- Composition of affine isometries. -/
def comp (g : P₂ →ᵃⁱ[𝕜] P₃) (f : P →ᵃⁱ[𝕜] P₂) : P →ᵃⁱ[𝕜] P₃ :=
  ⟨g.toAffineMap.comp f.toAffineMap, fun _ => (g.norm_map _).trans (f.norm_map _)⟩


@[simp]
theorem coe_comp (g : P₂ →ᵃⁱ[𝕜] P₃) (f : P →ᵃⁱ[𝕜] P₂) : ⇑(g.comp f) = g ∘ f :=
  rfl


@[simp]
theorem id_comp : (id : P₂ →ᵃⁱ[𝕜] P₂).comp f = f :=
  ext fun _ => rfl


@[simp]
theorem comp_id : f.comp id = f :=
  ext fun _ => rfl


theorem comp_assoc (f : P₃ →ᵃⁱ[𝕜] P₄) (g : P₂ →ᵃⁱ[𝕜] P₃) (h : P →ᵃⁱ[𝕜] P₂) :
    (f.comp g).comp h = f.comp (g.comp h) :=
  rfl


instance : Monoid (P →ᵃⁱ[𝕜] P) where
  one := id
  mul := comp
  mul_assoc := comp_assoc
  one_mul := id_comp
  mul_one := comp_id


@[simp]
theorem coe_one : ⇑(1 : P →ᵃⁱ[𝕜] P) = _root_.id :=
  rfl


@[simp]
theorem coe_mul (f g : P →ᵃⁱ[𝕜] P) : ⇑(f * g) = f ∘ g :=
  rfl


/-- `AffineSubspace.subtype` as an `AffineIsometry`. -/
def subtypeₐᵢ (s : AffineSubspace 𝕜 P) [Nonempty s] : s →ᵃⁱ[𝕜] P :=
  { s.subtype with norm_map := s.direction.subtypeₗᵢ.norm_map }


theorem subtypeₐᵢ_linear (s : AffineSubspace 𝕜 P) [Nonempty s] :
    s.subtypeₐᵢ.linear = s.direction.subtype :=
  rfl


@[simp]
theorem subtypeₐᵢ_linearIsometry (s : AffineSubspace 𝕜 P) [Nonempty s] :
    s.subtypeₐᵢ.linearIsometry = s.direction.subtypeₗᵢ :=
  rfl


@[simp]
theorem coe_subtypeₐᵢ (s : AffineSubspace 𝕜 P) [Nonempty s] : ⇑s.subtypeₐᵢ = s.subtype :=
  rfl


@[simp]
theorem subtypeₐᵢ_toAffineMap (s : AffineSubspace 𝕜 P) [Nonempty s] :
    s.subtypeₐᵢ.toAffineMap = s.subtype :=
  rfl


/-- An affine isometric equivalence between two normed vector spaces. -/
structure AffineIsometryEquiv extends P ≃ᵃ[𝕜] P₂ where
  norm_map : ∀ x, ‖linear x‖ = ‖x‖


@[inherit_doc] notation:25 P " ≃ᵃⁱ[" 𝕜:25 "] " P₂:0 => AffineIsometryEquiv 𝕜 P P₂


/-- The underlying linear equiv of an affine isometry equiv is in fact a linear isometry equiv. -/
protected def linearIsometryEquiv : V ≃ₗᵢ[𝕜] V₂ :=
  { e.linear with norm_map' := e.norm_map }


@[simp]
theorem linear_eq_linear_isometry : e.linear = e.linearIsometryEquiv.toLinearEquiv := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    e : AffineIsometryEquiv 𝕜 P P₂
    ⊢ Eq e.linear e.linearIsometryEquiv.toLinearEquiv
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    e : AffineIsometryEquiv 𝕜 P P₂
    x✝ : V
    ⊢ Eq (e.linear x✝) (e.linearIsometryEquiv.toLinearEquiv x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : EquivLike (P ≃ᵃⁱ[𝕜] P₂) P P₂ where
  coe f := f.toFun
  inv f := f.invFun
  left_inv f := f.left_inv
  right_inv f := f.right_inv
  coe_injective' f g h _ := by
    /-
      𝕜 : Type u_1
      V : Type u_2
      V₁ : Type u_3
      V₁' : Type u_4
      V₂ : Type u_5
      V₃ : Type u_6
      V₄ : Type u_7
      P₁ : Type u_8
      P₁' : Type u_9
      P : Type u_10
      P₂ : Type u_11
      P₃ : Type u_12
      P₄ : Type u_13
      inst✝²⁴ : NormedField 𝕜
      inst✝²³ : SeminormedAddCommGroup V
      inst✝²² : NormedSpace 𝕜 V
      inst✝²¹ : PseudoMetricSpace P
      inst✝²⁰ : NormedAddTorsor V P
      inst✝¹⁹ : SeminormedAddCommGroup V₁
      inst✝¹⁸ : NormedSpace 𝕜 V₁
      inst✝¹⁷ : PseudoMetricSpace P₁
      inst✝¹⁶ : NormedAddTorsor V₁ P₁
      inst✝¹⁵ : SeminormedAddCommGroup V₁'
      inst✝¹⁴ : NormedSpace 𝕜 V₁'
      inst✝¹³ : MetricSpace P₁'
      inst✝¹² : NormedAddTorsor V₁' P₁'
      inst✝¹¹ : SeminormedAddCommGroup V₂
      inst✝¹⁰ : NormedSpace 𝕜 V₂
      inst✝⁹ : PseudoMetricSpace P₂
      inst✝⁸ : NormedAddTorsor V₂ P₂
      inst✝⁷ : SeminormedAddCommGroup V₃
      inst✝⁶ : NormedSpace 𝕜 V₃
      inst✝⁵ : PseudoMetricSpace P₃
      inst✝⁴ : NormedAddTorsor V₃ P₃
      inst✝³ : SeminormedAddCommGroup V₄
      inst✝² : NormedSpace 𝕜 V₄
      inst✝¹ : PseudoMetricSpace P₄
      inst✝ : NormedAddTorsor V₄ P₄
      e f g : AffineIsometryEquiv 𝕜 P P₂
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      x✝ : Eq ((fun f => f.invFun) f) ((fun f => f.invFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      𝕜 : Type u_1
      V : Type u_2
      V₁ : Type u_3
      V₁' : Type u_4
      V₂ : Type u_5
      V₃ : Type u_6
      V₄ : Type u_7
      P₁ : Type u_8
      P₁' : Type u_9
      P : Type u_10
      P₂ : Type u_11
      P₃ : Type u_12
      P₄ : Type u_13
      inst✝²⁴ : NormedField 𝕜
      inst✝²³ : SeminormedAddCommGroup V
      inst✝²² : NormedSpace 𝕜 V
      inst✝²¹ : PseudoMetricSpace P
      inst✝²⁰ : NormedAddTorsor V P
      inst✝¹⁹ : SeminormedAddCommGroup V₁
      inst✝¹⁸ : NormedSpace 𝕜 V₁
      inst✝¹⁷ : PseudoMetricSpace P₁
      inst✝¹⁶ : NormedAddTorsor V₁ P₁
      inst✝¹⁵ : SeminormedAddCommGroup V₁'
      inst✝¹⁴ : NormedSpace 𝕜 V₁'
      inst✝¹³ : MetricSpace P₁'
      inst✝¹² : NormedAddTorsor V₁' P₁'
      inst✝¹¹ : SeminormedAddCommGroup V₂
      inst✝¹⁰ : NormedSpace 𝕜 V₂
      inst✝⁹ : PseudoMetricSpace P₂
      inst✝⁸ : NormedAddTorsor V₂ P₂
      inst✝⁷ : SeminormedAddCommGroup V₃
      inst✝⁶ : NormedSpace 𝕜 V₃
      inst✝⁵ : PseudoMetricSpace P₃
      inst✝⁴ : NormedAddTorsor V₃ P₃
      inst✝³ : SeminormedAddCommGroup V₄
      inst✝² : NormedSpace 𝕜 V₄
      inst✝¹ : PseudoMetricSpace P₄
      inst✝ : NormedAddTorsor V₄ P₄
      e g : AffineIsometryEquiv 𝕜 P P₂
      toAffineEquiv✝ : AffineEquiv 𝕜 P P₂
      norm_map✝ : ∀ (x : V), Eq (Norm.norm (toAffineEquiv✝.linear x)) (Norm.norm x)
      h : Eq ((fun f => f.toFun) { toAffineEquiv := toAffineEquiv✝, norm_map := norm …
      x✝ : Eq ((fun f => f.invFun) { toAffineEquiv := toAffineEquiv✝, norm_map := no …
      ⊢ Eq { toAffineEquiv := toAffineEquiv✝, norm_map := norm_map✝ } g
    -/
    cases g
    /-
      case mk.mk
      𝕜 : Type u_1
      V : Type u_2
      V₁ : Type u_3
      V₁' : Type u_4
      V₂ : Type u_5
      V₃ : Type u_6
      V₄ : Type u_7
      P₁ : Type u_8
      P₁' : Type u_9
      P : Type u_10
      P₂ : Type u_11
      P₃ : Type u_12
      P₄ : Type u_13
      inst✝²⁴ : NormedField 𝕜
      inst✝²³ : SeminormedAddCommGroup V
      inst✝²² : NormedSpace 𝕜 V
      inst✝²¹ : PseudoMetricSpace P
      inst✝²⁰ : NormedAddTorsor V P
      inst✝¹⁹ : SeminormedAddCommGroup V₁
      inst✝¹⁸ : NormedSpace 𝕜 V₁
      inst✝¹⁷ : PseudoMetricSpace P₁
      inst✝¹⁶ : NormedAddTorsor V₁ P₁
      inst✝¹⁵ : SeminormedAddCommGroup V₁'
      inst✝¹⁴ : NormedSpace 𝕜 V₁'
      inst✝¹³ : MetricSpace P₁'
      inst✝¹² : NormedAddTorsor V₁' P₁'
      inst✝¹¹ : SeminormedAddCommGroup V₂
      inst✝¹⁰ : NormedSpace 𝕜 V₂
      inst✝⁹ : PseudoMetricSpace P₂
      inst✝⁸ : NormedAddTorsor V₂ P₂
      inst✝⁷ : SeminormedAddCommGroup V₃
      inst✝⁶ : NormedSpace 𝕜 V₃
      inst✝⁵ : PseudoMetricSpace P₃
      inst✝⁴ : NormedAddTorsor V₃ P₃
      inst✝³ : SeminormedAddCommGroup V₄
      inst✝² : NormedSpace 𝕜 V₄
      inst✝¹ : PseudoMetricSpace P₄
      inst✝ : NormedAddTorsor V₄ P₄
      e : AffineIsometryEquiv 𝕜 P P₂
      toAffineEquiv✝¹ : AffineEquiv 𝕜 P P₂
      norm_map✝¹ : ∀ (x : V), Eq (Norm.norm (toAffineEquiv✝¹.linear x)) (Norm.norm x)
      toAffineEquiv✝ : AffineEquiv 𝕜 P P₂
      norm_map✝ : ∀ (x : V), Eq (Norm.norm (toAffineEquiv✝.linear x)) (Norm.norm x)
      h : Eq ((fun f => f.toFun) { toAffineEquiv := toAffineEquiv✝¹, norm_map := nor …
      x✝ : Eq ((fun f => f.invFun) { toAffineEquiv := toAffineEquiv✝¹, norm_map := n …
      ⊢ Eq { toAffineEquiv := toAffineEquiv✝¹, norm_map := norm_map✝¹ } { toAffineEq …
    -/
    congr
    /-
      case mk.mk.e_toAffineEquiv
      𝕜 : Type u_1
      V : Type u_2
      V₁ : Type u_3
      V₁' : Type u_4
      V₂ : Type u_5
      V₃ : Type u_6
      V₄ : Type u_7
      P₁ : Type u_8
      P₁' : Type u_9
      P : Type u_10
      P₂ : Type u_11
      P₃ : Type u_12
      P₄ : Type u_13
      inst✝²⁴ : NormedField 𝕜
      inst✝²³ : SeminormedAddCommGroup V
      inst✝²² : NormedSpace 𝕜 V
      inst✝²¹ : PseudoMetricSpace P
      inst✝²⁰ : NormedAddTorsor V P
      inst✝¹⁹ : SeminormedAddCommGroup V₁
      inst✝¹⁸ : NormedSpace 𝕜 V₁
      inst✝¹⁷ : PseudoMetricSpace P₁
      inst✝¹⁶ : NormedAddTorsor V₁ P₁
      inst✝¹⁵ : SeminormedAddCommGroup V₁'
      inst✝¹⁴ : NormedSpace 𝕜 V₁'
      inst✝¹³ : MetricSpace P₁'
      inst✝¹² : NormedAddTorsor V₁' P₁'
      inst✝¹¹ : SeminormedAddCommGroup V₂
      inst✝¹⁰ : NormedSpace 𝕜 V₂
      inst✝⁹ : PseudoMetricSpace P₂
      inst✝⁸ : NormedAddTorsor V₂ P₂
      inst✝⁷ : SeminormedAddCommGroup V₃
      inst✝⁶ : NormedSpace 𝕜 V₃
      inst✝⁵ : PseudoMetricSpace P₃
      inst✝⁴ : NormedAddTorsor V₃ P₃
      inst✝³ : SeminormedAddCommGroup V₄
      inst✝² : NormedSpace 𝕜 V₄
      inst✝¹ : PseudoMetricSpace P₄
      inst✝ : NormedAddTorsor V₄ P₄
      e : AffineIsometryEquiv 𝕜 P P₂
      toAffineEquiv✝¹ : AffineEquiv 𝕜 P P₂
      norm_map✝¹ : ∀ (x : V), Eq (Norm.norm (toAffineEquiv✝¹.linear x)) (Norm.norm x)
      toAffineEquiv✝ : AffineEquiv 𝕜 P P₂
      norm_map✝ : ∀ (x : V), Eq (Norm.norm (toAffineEquiv✝.linear x)) (Norm.norm x)
      h : Eq ((fun f => f.toFun) { toAffineEquiv := toAffineEquiv✝¹, norm_map := nor …
      x✝ : Eq ((fun f => f.invFun) { toAffineEquiv := toAffineEquiv✝¹, norm_map := n …
      ⊢ Eq toAffineEquiv✝¹ toAffineEquiv✝
    -/
    simpa [DFunLike.coe_injective.eq_iff] using h
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_mk (e : P ≃ᵃ[𝕜] P₂) (he : ∀ x, ‖e.linear x‖ = ‖x‖) : ⇑(mk e he) = e :=
  rfl


@[simp]
theorem coe_toAffineEquiv (e : P ≃ᵃⁱ[𝕜] P₂) : ⇑e.toAffineEquiv = e :=
  rfl


theorem toAffineEquiv_injective : Injective (toAffineEquiv : (P ≃ᵃⁱ[𝕜] P₂) → P ≃ᵃ[𝕜] P₂)
  | ⟨_, _⟩, ⟨_, _⟩, rfl => rfl


@[ext]
theorem ext {e e' : P ≃ᵃⁱ[𝕜] P₂} (h : ∀ x, e x = e' x) : e = e' :=
  toAffineEquiv_injective <| AffineEquiv.ext h


/-- Reinterpret an `AffineIsometryEquiv` as an `AffineIsometry`. -/
def toAffineIsometry : P →ᵃⁱ[𝕜] P₂ :=
  ⟨e.1.toAffineMap, e.2⟩


@[simp]
theorem coe_toAffineIsometry : ⇑e.toAffineIsometry = e :=
  rfl


/-- Construct an affine isometry equivalence by verifying the relation between the map and its
linear part at one base point. Namely, this function takes a map `e : P₁ → P₂`, a linear isometry
equivalence `e' : V₁ ≃ᵢₗ[k] V₂`, and a point `p` such that for any other point `p'` we have
`e p' = e' (p' -ᵥ p) +ᵥ e p`. -/
def mk' (e : P₁ → P₂) (e' : V₁ ≃ₗᵢ[𝕜] V₂) (p : P₁) (h : ∀ p' : P₁, e p' = e' (p' -ᵥ p) +ᵥ e p) :
    P₁ ≃ᵃⁱ[𝕜] P₂ :=
  { AffineEquiv.mk' e e'.toLinearEquiv p h with norm_map := e'.norm_map }


@[simp]
theorem coe_mk' (e : P₁ → P₂) (e' : V₁ ≃ₗᵢ[𝕜] V₂) (p h) : ⇑(mk' e e' p h) = e :=
  rfl


@[simp]
theorem linearIsometryEquiv_mk' (e : P₁ → P₂) (e' : V₁ ≃ₗᵢ[𝕜] V₂) (p h) :
    (mk' e e' p h).linearIsometryEquiv = e' := by
  /-
    𝕜 : Type u_1
    V₁ : Type u_3
    V₂ : Type u_5
    P₁ : Type u_8
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V₁
    inst✝⁶ : NormedSpace 𝕜 V₁
    inst✝⁵ : PseudoMetricSpace P₁
    inst✝⁴ : NormedAddTorsor V₁ P₁
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    e : P₁ → P₂
    e' : LinearIsometryEquiv (RingHom.id 𝕜) V₁ V₂
    p : P₁
    h : ∀ (p' : P₁), Eq (e p') (HVAdd.hVAdd (e' (VSub.vsub p' p)) (e p))
    ⊢ Eq (AffineIsometryEquiv.mk' e e' p h).linearIsometryEquiv e'
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    V₁ : Type u_3
    V₂ : Type u_5
    P₁ : Type u_8
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V₁
    inst✝⁶ : NormedSpace 𝕜 V₁
    inst✝⁵ : PseudoMetricSpace P₁
    inst✝⁴ : NormedAddTorsor V₁ P₁
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    e : P₁ → P₂
    e' : LinearIsometryEquiv (RingHom.id 𝕜) V₁ V₂
    p : P₁
    h : ∀ (p' : P₁), Eq (e p') (HVAdd.hVAdd (e' (VSub.vsub p' p)) (e p))
    x✝ : V₁
    ⊢ Eq ((AffineIsometryEquiv.mk' e e' p h).linearIsometryEquiv x✝) (e' x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Reinterpret a linear isometry equiv as an affine isometry equiv. -/
def toAffineIsometryEquiv : V ≃ᵃⁱ[𝕜] V₂ :=
  { e.toLinearEquiv.toAffineEquiv with norm_map := e.norm_map }


@[simp]
theorem coe_toAffineIsometryEquiv : ⇑(e.toAffineIsometryEquiv : V ≃ᵃⁱ[𝕜] V₂) = e := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : NormedSpace 𝕜 V₂
    e : LinearIsometryEquiv (RingHom.id 𝕜) V V₂
    ⊢ Eq ⇑e.toAffineIsometryEquiv ⇑e
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem toAffineIsometryEquiv_linearIsometryEquiv :
    e.toAffineIsometryEquiv.linearIsometryEquiv = e := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : NormedSpace 𝕜 V₂
    e : LinearIsometryEquiv (RingHom.id 𝕜) V V₂
    ⊢ Eq e.toAffineIsometryEquiv.linearIsometryEquiv e
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : SeminormedAddCommGroup V₂
    inst✝ : NormedSpace 𝕜 V₂
    e : LinearIsometryEquiv (RingHom.id 𝕜) V V₂
    x✝ : V
    ⊢ Eq (e.toAffineIsometryEquiv.linearIsometryEquiv x✝) (e x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/

-- somewhat arbitrary choice of simp direction

@[simp]
theorem toAffineIsometryEquiv_toAffineEquiv :
    e.toAffineIsometryEquiv.toAffineEquiv = e.toLinearEquiv.toAffineEquiv :=
  rfl

-- somewhat arbitrary choice of simp direction

@[simp]
theorem toAffineIsometryEquiv_toAffineIsometry :
    e.toAffineIsometryEquiv.toAffineIsometry = e.toLinearIsometry.toAffineIsometry :=
  rfl


protected theorem isometry : Isometry e :=
  e.toAffineIsometry.isometry


/-- Reinterpret an `AffineIsometryEquiv` as an `IsometryEquiv`. -/
def toIsometryEquiv : P ≃ᵢ P₂ :=
  ⟨e.toAffineEquiv.toEquiv, e.isometry⟩


@[simp]
theorem coe_toIsometryEquiv : ⇑e.toIsometryEquiv = e :=
  rfl


theorem range_eq_univ (e : P ≃ᵃⁱ[𝕜] P₂) : Set.range e = Set.univ := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    e : AffineIsometryEquiv 𝕜 P P₂
    ⊢ Eq (Set.range ⇑e) Set.univ
  -/
  rw [← coe_toIsometryEquiv]
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    e : AffineIsometryEquiv 𝕜 P P₂
    ⊢ Eq (Set.range ⇑e.toIsometryEquiv) Set.univ
  -/
  exact IsometryEquiv.range_eq_univ _
  /-
    🎉 no goals
  -/


/-- Reinterpret an `AffineIsometryEquiv` as a `Homeomorph`. -/
def toHomeomorph : P ≃ₜ P₂ :=
  e.toIsometryEquiv.toHomeomorph


@[simp]
theorem coe_toHomeomorph : ⇑e.toHomeomorph = e :=
  rfl


protected theorem continuous : Continuous e :=
  e.isometry.continuous


protected theorem continuousAt {x} : ContinuousAt e x :=
  e.continuous.continuousAt


protected theorem continuousOn {s} : ContinuousOn e s :=
  e.continuous.continuousOn


protected theorem continuousWithinAt {s x} : ContinuousWithinAt e s x :=
  e.continuous.continuousWithinAt


/-- Identity map as an `AffineIsometryEquiv`. -/
def refl : P ≃ᵃⁱ[𝕜] P :=
  ⟨AffineEquiv.refl 𝕜 P, fun _ => rfl⟩


instance : Inhabited (P ≃ᵃⁱ[𝕜] P) :=
  ⟨refl 𝕜 P⟩


@[simp]
theorem coe_refl : ⇑(refl 𝕜 P) = id :=
  rfl


@[simp]
theorem toAffineEquiv_refl : (refl 𝕜 P).toAffineEquiv = AffineEquiv.refl 𝕜 P :=
  rfl


@[simp]
theorem toIsometryEquiv_refl : (refl 𝕜 P).toIsometryEquiv = IsometryEquiv.refl P :=
  rfl


@[simp]
theorem toHomeomorph_refl : (refl 𝕜 P).toHomeomorph = Homeomorph.refl P :=
  rfl


/-- The inverse `AffineIsometryEquiv`. -/
def symm : P₂ ≃ᵃⁱ[𝕜] P :=
  { e.toAffineEquiv.symm with norm_map := e.linearIsometryEquiv.symm.norm_map }


@[simp]
theorem apply_symm_apply (x : P₂) : e (e.symm x) = x :=
  e.toAffineEquiv.apply_symm_apply x


@[simp]
theorem symm_apply_apply (x : P) : e.symm (e x) = x :=
  e.toAffineEquiv.symm_apply_apply x


@[simp]
theorem symm_symm : e.symm.symm = e := rfl


@[simp]
theorem toAffineEquiv_symm : e.toAffineEquiv.symm = e.symm.toAffineEquiv :=
  rfl


@[simp]
theorem toIsometryEquiv_symm : e.toIsometryEquiv.symm = e.symm.toIsometryEquiv :=
  rfl


@[simp]
theorem toHomeomorph_symm : e.toHomeomorph.symm = e.symm.toHomeomorph :=
  rfl


/-- Composition of `AffineIsometryEquiv`s as an `AffineIsometryEquiv`. -/
def trans (e' : P₂ ≃ᵃⁱ[𝕜] P₃) : P ≃ᵃⁱ[𝕜] P₃ :=
  ⟨e.toAffineEquiv.trans e'.toAffineEquiv, fun _ => (e'.norm_map _).trans (e.norm_map _)⟩


@[simp]
theorem coe_trans (e₁ : P ≃ᵃⁱ[𝕜] P₂) (e₂ : P₂ ≃ᵃⁱ[𝕜] P₃) : ⇑(e₁.trans e₂) = e₂ ∘ e₁ :=
  rfl


@[simp]
theorem trans_refl : e.trans (refl 𝕜 P₂) = e :=
  ext fun _ => rfl


@[simp]
theorem refl_trans : (refl 𝕜 P).trans e = e :=
  ext fun _ => rfl


@[simp]
theorem self_trans_symm : e.trans e.symm = refl 𝕜 P :=
  ext e.symm_apply_apply


@[simp]
theorem symm_trans_self : e.symm.trans e = refl 𝕜 P₂ :=
  ext e.apply_symm_apply


@[simp]
theorem coe_symm_trans (e₁ : P ≃ᵃⁱ[𝕜] P₂) (e₂ : P₂ ≃ᵃⁱ[𝕜] P₃) :
    ⇑(e₁.trans e₂).symm = e₁.symm ∘ e₂.symm :=
  rfl


theorem trans_assoc (ePP₂ : P ≃ᵃⁱ[𝕜] P₂) (eP₂G : P₂ ≃ᵃⁱ[𝕜] P₃) (eGG' : P₃ ≃ᵃⁱ[𝕜] P₄) :
    ePP₂.trans (eP₂G.trans eGG') = (ePP₂.trans eP₂G).trans eGG' :=
  rfl


/-- The group of affine isometries of a `NormedAddTorsor`, `P`. -/
instance instGroup : Group (P ≃ᵃⁱ[𝕜] P) where
  mul e₁ e₂ := e₂.trans e₁
  one := refl _ _
  inv := symm
  one_mul := trans_refl
  mul_one := refl_trans
  mul_assoc _ _ _ := trans_assoc _ _ _
  inv_mul_cancel := self_trans_symm


@[simp]
theorem coe_one : ⇑(1 : P ≃ᵃⁱ[𝕜] P) = id :=
  rfl


@[simp]
theorem coe_mul (e e' : P ≃ᵃⁱ[𝕜] P) : ⇑(e * e') = e ∘ e' :=
  rfl


@[simp]
theorem coe_inv (e : P ≃ᵃⁱ[𝕜] P) : ⇑e⁻¹ = e.symm :=
  rfl


@[simp]
theorem map_vadd (p : P) (v : V) : e (v +ᵥ p) = e.linearIsometryEquiv v +ᵥ e p :=
  e.toAffineIsometry.map_vadd p v


@[simp]
theorem map_vsub (p1 p2 : P) : e.linearIsometryEquiv (p1 -ᵥ p2) = e p1 -ᵥ e p2 :=
  e.toAffineIsometry.map_vsub p1 p2


@[simp]
theorem dist_map (x y : P) : dist (e x) (e y) = dist x y :=
  e.toAffineIsometry.dist_map x y


@[simp]
theorem edist_map (x y : P) : edist (e x) (e y) = edist x y :=
  e.toAffineIsometry.edist_map x y


protected theorem bijective : Bijective e :=
  e.1.bijective


protected theorem injective : Injective e :=
  e.1.injective


protected theorem surjective : Surjective e :=
  e.1.surjective


theorem map_eq_iff {x y : P} : e x = e y ↔ x = y :=
  e.injective.eq_iff


theorem map_ne {x y : P} (h : x ≠ y) : e x ≠ e y :=
  e.injective.ne h


protected theorem lipschitz : LipschitzWith 1 e :=
  e.isometry.lipschitz


protected theorem antilipschitz : AntilipschitzWith 1 e :=
  e.isometry.antilipschitz


@[simp]
theorem ediam_image (s : Set P) : EMetric.diam (e '' s) = EMetric.diam s :=
  e.isometry.ediam_image s


@[simp]
theorem diam_image (s : Set P) : Metric.diam (e '' s) = Metric.diam s :=
  e.isometry.diam_image s


@[simp]
theorem comp_continuousOn_iff {f : α → P} {s : Set α} : ContinuousOn (e ∘ f) s ↔ ContinuousOn f s :=
  e.isometry.comp_continuousOn_iff


@[simp]
theorem comp_continuous_iff {f : α → P} : Continuous (e ∘ f) ↔ Continuous f :=
  e.isometry.comp_continuous_iff


/-- The map `v ↦ v +ᵥ p` as an affine isometric equivalence between `V` and `P`. -/
def vaddConst (p : P) : V ≃ᵃⁱ[𝕜] P :=
  { AffineEquiv.vaddConst 𝕜 p with norm_map := fun _ => rfl }


@[simp]
theorem coe_vaddConst (p : P) : ⇑(vaddConst 𝕜 p) = fun v => v +ᵥ p :=
  rfl


@[simp]
theorem coe_vaddConst' (p : P) : ↑(AffineEquiv.vaddConst 𝕜 p) = fun v => v +ᵥ p :=
  rfl


@[simp]
theorem coe_vaddConst_symm (p : P) : ⇑(vaddConst 𝕜 p).symm = fun p' => p' -ᵥ p :=
  rfl


@[simp]
theorem vaddConst_toAffineEquiv (p : P) :
    (vaddConst 𝕜 p).toAffineEquiv = AffineEquiv.vaddConst 𝕜 p :=
  rfl


/-- `p' ↦ p -ᵥ p'` as an affine isometric equivalence. -/
def constVSub (p : P) : P ≃ᵃⁱ[𝕜] V :=
  { AffineEquiv.constVSub 𝕜 p with norm_map := norm_neg }


@[simp]
theorem coe_constVSub (p : P) : ⇑(constVSub 𝕜 p) = (p -ᵥ ·) :=
  rfl


@[simp]
theorem symm_constVSub (p : P) :
    (constVSub 𝕜 p).symm =
      (LinearIsometryEquiv.neg 𝕜).toAffineIsometryEquiv.trans (vaddConst 𝕜 p) := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_10
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p : P
    ⊢ Eq (AffineIsometryEquiv.constVSub 𝕜 p).symm ((LinearIsometryEquiv.neg 𝕜).toA …
  -/
  ext
  /-
    case h
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_10
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    p : P
    x✝ : V
    ⊢ Eq ((AffineIsometryEquiv.constVSub 𝕜 p).symm x✝) (((LinearIsometryEquiv.neg  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Translation by `v` (that is, the map `p ↦ v +ᵥ p`) as an affine isometric automorphism of `P`.
-/
def constVAdd (v : V) : P ≃ᵃⁱ[𝕜] P :=
  { AffineEquiv.constVAdd 𝕜 P v with norm_map := fun _ => rfl }


@[simp]
theorem coe_constVAdd (v : V) : ⇑(constVAdd 𝕜 P v : P ≃ᵃⁱ[𝕜] P) = (v +ᵥ ·) :=
  rfl


@[simp]
theorem constVAdd_zero : constVAdd 𝕜 P (0 : V) = refl 𝕜 P :=
  ext <| zero_vadd V


include 𝕜 in
/-- The map `g` from `V` to `V₂` corresponding to a map `f` from `P` to `P₂`, at a base point `p`,
is an isometry if `f` is one. -/
theorem vadd_vsub {f : P → P₂} (hf : Isometry f) {p : P} {g : V → V₂}
    (hg : ∀ v, g v = f (v +ᵥ p) -ᵥ f p) : Isometry g := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : P → P₂
    hf : Isometry f
    p : P
    g : V → V₂
    hg : ∀ (v : V), Eq (g v) (VSub.vsub (f (HVAdd.hVAdd v p)) (f p))
    ⊢ Isometry g
  -/
  convert (vaddConst 𝕜 (f p)).symm.isometry.comp (hf.comp (vaddConst 𝕜 p).isometry)
  /-
    case h.e'_5
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : P → P₂
    hf : Isometry f
    p : P
    g : V → V₂
    hg : ∀ (v : V), Eq (g v) (VSub.vsub (f (HVAdd.hVAdd v p)) (f p))
    ⊢ Eq g (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f p)).symm) (Functio …
  -/
  exact funext hg
  /-
    🎉 no goals
  -/


/-- Point reflection in `x` as an affine isometric automorphism. -/
def pointReflection (x : P) : P ≃ᵃⁱ[𝕜] P :=
  (constVSub 𝕜 x).trans (vaddConst 𝕜 x)


theorem pointReflection_apply (x y : P) : (pointReflection 𝕜 x) y = (x -ᵥ y) +ᵥ x :=
  rfl


@[simp]
theorem pointReflection_toAffineEquiv (x : P) :
    (pointReflection 𝕜 x).toAffineEquiv = AffineEquiv.pointReflection 𝕜 x :=
  rfl


@[simp]
theorem pointReflection_self (x : P) : pointReflection 𝕜 x x = x :=
  AffineEquiv.pointReflection_self 𝕜 x


theorem pointReflection_involutive (x : P) : Function.Involutive (pointReflection 𝕜 x) :=
  Equiv.pointReflection_involutive x


@[simp]
theorem pointReflection_symm (x : P) : (pointReflection 𝕜 x).symm = pointReflection 𝕜 x :=
  toAffineEquiv_injective <| AffineEquiv.pointReflection_symm 𝕜 x


@[simp]
theorem dist_pointReflection_fixed (x y : P) : dist (pointReflection 𝕜 x y) x = dist y x := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_10
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    x y : P
    ⊢ Eq (Dist.dist ((AffineIsometryEquiv.pointReflection 𝕜 x) y) x) (Dist.dist y x)
  -/
  rw [← (pointReflection 𝕜 x).dist_map y x, pointReflection_self]
  /-
    🎉 no goals
  -/


theorem dist_pointReflection_self' (x y : P) :
    dist (pointReflection 𝕜 x y) y = ‖2 • (x -ᵥ y)‖ := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_10
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    x y : P
    ⊢ Eq (Dist.dist ((AffineIsometryEquiv.pointReflection 𝕜 x) y) y) (Norm.norm (H …
  -/
  rw [pointReflection_apply, dist_eq_norm_vsub V, vadd_vsub_assoc, two_nsmul]
  /-
    🎉 no goals
  -/


theorem dist_pointReflection_self (x y : P) :
    dist (pointReflection 𝕜 x y) y = ‖(2 : 𝕜)‖ * dist x y := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    P : Type u_10
    inst✝⁴ : NormedField 𝕜
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : NormedSpace 𝕜 V
    inst✝¹ : PseudoMetricSpace P
    inst✝ : NormedAddTorsor V P
    x y : P
    ⊢ Eq (Dist.dist ((AffineIsometryEquiv.pointReflection 𝕜 x) y) y) (HMul.hMul (N …
  -/
  rw [dist_pointReflection_self', two_nsmul, ← two_smul 𝕜, norm_smul, ← dist_eq_norm_vsub V]
  /-
    🎉 no goals
  -/


theorem pointReflection_fixed_iff [Invertible (2 : 𝕜)] {x y : P} :
    pointReflection 𝕜 x y = y ↔ y = x :=
  AffineEquiv.pointReflection_fixed_iff_of_module 𝕜


theorem dist_pointReflection_self_real (x y : P) :
    dist (pointReflection ℝ x y) y = 2 * dist x y := by
  /-
    V : Type u_2
    P : Type u_10
    inst✝³ : SeminormedAddCommGroup V
    inst✝² : PseudoMetricSpace P
    inst✝¹ : NormedAddTorsor V P
    inst✝ : NormedSpace Real V
    x y : P
    ⊢ Eq (Dist.dist ((AffineIsometryEquiv.pointReflection Real x) y) y) (HMul.hMul …
  -/
  rw [dist_pointReflection_self, Real.norm_two]
  /-
    🎉 no goals
  -/


@[simp]
theorem pointReflection_midpoint_left (x y : P) : pointReflection ℝ (midpoint ℝ x y) x = y :=
  AffineEquiv.pointReflection_midpoint_left x y


@[simp]
theorem pointReflection_midpoint_right (x y : P) : pointReflection ℝ (midpoint ℝ x y) y = x :=
  AffineEquiv.pointReflection_midpoint_right x y


/-- If `f` is an affine map, then its linear part is continuous iff `f` is continuous. -/
theorem AffineMap.continuous_linear_iff {f : P →ᵃ[𝕜] P₂} : Continuous f.linear ↔ Continuous f := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    ⊢ Iff (Continuous ⇑f.linear) (Continuous ⇑f)
  -/
  inhabit P
  have :
    (f.linear : V → V₂) =
      (AffineIsometryEquiv.vaddConst 𝕜 <| f default).toHomeomorph.symm ∘
        f ∘ (AffineIsometryEquiv.vaddConst 𝕜 default).toHomeomorph := by
    ext v
    simp
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    inhabited_h : Inhabited P
    this : Eq (⇑f.linear) (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f Inh …
    ⊢ Iff (Continuous ⇑f.linear) (Continuous ⇑f)
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    inhabited_h : Inhabited P
    this : Eq (⇑f.linear) (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f Inh …
    ⊢ Iff (Continuous (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f Inhabit …
  -/
  simp only [Homeomorph.comp_continuous_iff, Homeomorph.comp_continuous_iff']
  /-
    🎉 no goals
  -/


/-- If `f` is an affine map, then its linear part is an open map iff `f` is an open map. -/
theorem AffineMap.isOpenMap_linear_iff {f : P →ᵃ[𝕜] P₂} : IsOpenMap f.linear ↔ IsOpenMap f := by
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    ⊢ Iff (IsOpenMap ⇑f.linear) (IsOpenMap ⇑f)
  -/
  inhabit P
  have :
    (f.linear : V → V₂) =
      (AffineIsometryEquiv.vaddConst 𝕜 <| f default).toHomeomorph.symm ∘
        f ∘ (AffineIsometryEquiv.vaddConst 𝕜 default).toHomeomorph := by
    ext v
    simp
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    inhabited_h : Inhabited P
    this : Eq (⇑f.linear) (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f Inh …
    ⊢ Iff (IsOpenMap ⇑f.linear) (IsOpenMap ⇑f)
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    V : Type u_2
    V₂ : Type u_5
    P : Type u_10
    P₂ : Type u_11
    inst✝⁸ : NormedField 𝕜
    inst✝⁷ : SeminormedAddCommGroup V
    inst✝⁶ : NormedSpace 𝕜 V
    inst✝⁵ : PseudoMetricSpace P
    inst✝⁴ : NormedAddTorsor V P
    inst✝³ : SeminormedAddCommGroup V₂
    inst✝² : NormedSpace 𝕜 V₂
    inst✝¹ : PseudoMetricSpace P₂
    inst✝ : NormedAddTorsor V₂ P₂
    f : AffineMap 𝕜 P P₂
    inhabited_h : Inhabited P
    this : Eq (⇑f.linear) (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f Inh …
    ⊢ Iff (IsOpenMap (Function.comp (⇑(AffineIsometryEquiv.vaddConst 𝕜 (f Inhabite …
  -/
  simp only [Homeomorph.comp_isOpenMap_iff, Homeomorph.comp_isOpenMap_iff']
  /-
    🎉 no goals
  -/


/-- An affine subspace is isomorphic to its image under an injective affine map.
This is the affine version of `Submodule.equivMapOfInjective`.
-/
@[simps linear, simps! toFun]
noncomputable def equivMapOfInjective (E : AffineSubspace 𝕜 P₁) [Nonempty E] (φ : P₁ →ᵃ[𝕜] P₂)
    (hφ : Function.Injective φ) : E ≃ᵃ[𝕜] E.map φ :=
  { Equiv.Set.image _ (E : Set P₁) hφ with
    linear :=
      (E.direction.equivMapOfInjective φ.linear (φ.linear_injective_iff.mpr hφ)).trans
        (LinearEquiv.ofEq _ _ (AffineSubspace.map_direction _ _).symm)
    map_vadd' := fun p v => Subtype.ext <| φ.map_vadd p v }


/-- Restricts an affine isometry to an affine isometry equivalence between a nonempty affine
subspace `E` and its image.

This is an isometry version of `AffineSubspace.equivMap`, having a stronger premise and a stronger
conclusion.
-/
noncomputable def isometryEquivMap (φ : P₁' →ᵃⁱ[𝕜] P₂) (E : AffineSubspace 𝕜 P₁') [Nonempty E] :
    E ≃ᵃⁱ[𝕜] E.map φ.toAffineMap :=
  ⟨E.equivMapOfInjective φ.toAffineMap φ.injective, fun _ => φ.norm_map _⟩


@[simp]
theorem isometryEquivMap.apply_symm_apply {E : AffineSubspace 𝕜 P₁'} [Nonempty E]
    {φ : P₁' →ᵃⁱ[𝕜] P₂} (x : E.map φ.toAffineMap) : φ ((E.isometryEquivMap φ).symm x) = x :=
  congr_arg Subtype.val <| (E.isometryEquivMap φ).apply_symm_apply _


@[simp]
theorem isometryEquivMap.coe_apply (φ : P₁' →ᵃⁱ[𝕜] P₂) (E : AffineSubspace 𝕜 P₁') [Nonempty E]
    (g : E) : ↑(E.isometryEquivMap φ g) = φ g :=
  rfl


@[simp]
theorem isometryEquivMap.toAffineMap_eq (φ : P₁' →ᵃⁱ[𝕜] P₂) (E : AffineSubspace 𝕜 P₁')
    [Nonempty E] :
    (E.isometryEquivMap φ).toAffineMap = E.equivMapOfInjective φ.toAffineMap φ.injective :=
  rfl


