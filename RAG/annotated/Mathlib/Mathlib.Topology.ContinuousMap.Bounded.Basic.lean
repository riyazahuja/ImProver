/-- `α →ᵇ β` is the type of bounded continuous functions `α → β` from a topological space to a
metric space.

When possible, instead of parametrizing results over `(f : α →ᵇ β)`,
you should parametrize over `(F : Type*) [BoundedContinuousMapClass F α β] (f : F)`.

When you extend this structure, make sure to extend `BoundedContinuousMapClass`. -/
structure BoundedContinuousFunction (α : Type u) (β : Type v) [TopologicalSpace α]
    [PseudoMetricSpace β] extends ContinuousMap α β : Type max u v where
  map_bounded' : ∃ C, ∀ x y, dist (toFun x) (toFun y) ≤ C


@[inherit_doc] scoped[BoundedContinuousFunction] infixr:25 " →ᵇ " => BoundedContinuousFunction


/-- `BoundedContinuousMapClass F α β` states that `F` is a type of bounded continuous maps.

You should also extend this typeclass when you extend `BoundedContinuousFunction`. -/
class BoundedContinuousMapClass (F : Type*) (α β : outParam Type*) [TopologicalSpace α]
    [PseudoMetricSpace β] [FunLike F α β] extends ContinuousMapClass F α β : Prop where
  map_bounded (f : F) : ∃ C, ∀ x y, dist (f x) (f y) ≤ C


instance instFunLike : FunLike (α →ᵇ β) α β where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : TopologicalSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : PseudoMetricSpace γ
      f✝ g✝ : BoundedContinuousFunction α β
      x : α
      C : Real
      f g : BoundedContinuousFunction α β
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    obtain ⟨⟨_, _⟩, _⟩ := f
    /-
      case mk.mk
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : TopologicalSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : PseudoMetricSpace γ
      f g✝ : BoundedContinuousFunction α β
      x : α
      C : Real
      g : BoundedContinuousFunction α β
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      map_bounded'✝ : Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := toFu …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝, continuous_toFun := continuous_t …
      ⊢ Eq { toFun := toFun✝, continuous_toFun := continuous_toFun✝, map_bounded' := …
    -/
    obtain ⟨⟨_, _⟩, _⟩ := g
    /-
      case mk.mk.mk.mk
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : TopologicalSpace α
      inst✝¹ : PseudoMetricSpace β
      inst✝ : PseudoMetricSpace γ
      f g : BoundedContinuousFunction α β
      x : α
      C : Real
      toFun✝¹ : α → β
      continuous_toFun✝¹ : Continuous toFun✝¹
      map_bounded'✝¹ : Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := toF …
      toFun✝ : α → β
      continuous_toFun✝ : Continuous toFun✝
      map_bounded'✝ : Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := toFu …
      h : Eq ((fun f => f.toFun) { toFun := toFun✝¹, continuous_toFun := continuous_ …
      ⊢ Eq { toFun := toFun✝¹, continuous_toFun := continuous_toFun✝¹, map_bounded'  …
    -/
    congr
    /-
      🎉 no goals
    -/


instance instBoundedContinuousMapClass : BoundedContinuousMapClass (α →ᵇ β) α β where
  map_continuous f := f.continuous_toFun
  map_bounded f := f.map_bounded'


instance instCoeTC [FunLike F α β] [BoundedContinuousMapClass F α β] : CoeTC F (α →ᵇ β) :=
  ⟨fun f =>
    { toFun := f
      continuous_toFun := map_continuous f
      map_bounded' := map_bounded f }⟩


@[simp]
theorem coe_toContinuousMap (f : α →ᵇ β) : (f.toContinuousMap : α → β) = f := rfl


@[deprecated (since := "2024-11-23")] alias coe_to_continuous_fun := coe_toContinuousMap


/-- See Note [custom simps projection]. We need to specify this projection explicitly in this case,
  because it is a composition of multiple projections. -/
def Simps.apply (h : α →ᵇ β) : α → β := h


protected theorem bounded (f : α →ᵇ β) : ∃ C, ∀ x y : α, dist (f x) (f y) ≤ C :=
  f.map_bounded'


protected theorem continuous (f : α →ᵇ β) : Continuous f :=
  f.toContinuousMap.continuous


@[ext]
theorem ext (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


theorem isBounded_range (f : α →ᵇ β) : IsBounded (range f) :=
  isBounded_range_iff.2 f.bounded


theorem isBounded_image (f : α →ᵇ β) (s : Set α) : IsBounded (f '' s) :=
  f.isBounded_range.subset <| image_subset_range _ _


theorem eq_of_empty [h : IsEmpty α] (f g : α →ᵇ β) : f = g :=
  ext <| h.elim


/-- A continuous function with an explicit bound is a bounded continuous function. -/
def mkOfBound (f : C(α, β)) (C : ℝ) (h : ∀ x y : α, dist (f x) (f y) ≤ C) : α →ᵇ β :=
  ⟨f, ⟨C, h⟩⟩


@[simp]
theorem mkOfBound_coe {f} {C} {h} : (mkOfBound f C h : α → β) = (f : α → β) := rfl


/-- A continuous function on a compact space is automatically a bounded continuous function. -/
def mkOfCompact [CompactSpace α] (f : C(α, β)) : α →ᵇ β :=
  ⟨f, isBounded_range_iff.1 (isCompact_range f.continuous).isBounded⟩


@[simp]
theorem mkOfCompact_apply [CompactSpace α] (f : C(α, β)) (a : α) : mkOfCompact f a = f a := rfl


/-- If a function is bounded on a discrete space, it is automatically continuous,
and therefore gives rise to an element of the type of bounded continuous functions. -/
@[simps]
def mkOfDiscrete [DiscreteTopology α] (f : α → β) (C : ℝ) (h : ∀ x y : α, dist (f x) (f y) ≤ C) :
    α →ᵇ β :=
  ⟨⟨f, continuous_of_discreteTopology⟩, ⟨C, h⟩⟩


/-- The uniform distance between two bounded continuous functions. -/
instance instDist : Dist (α →ᵇ β) :=
  ⟨fun f g => sInf { C | 0 ≤ C ∧ ∀ x : α, dist (f x) (g x) ≤ C }⟩


theorem dist_eq : dist f g = sInf { C | 0 ≤ C ∧ ∀ x : α, dist (f x) (g x) ≤ C } := rfl


theorem dist_set_exists : ∃ C, 0 ≤ C ∧ ∀ x : α, dist (f x) (g x) ≤ C := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    ⊢ Exists fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Dist.dist (f x) (g x)) C)
  -/
  rcases isBounded_iff.1 (f.isBounded_range.union g.isBounded_range) with ⟨C, hC⟩
  refine ⟨max 0 C, le_max_left _ _, fun x => (hC ?_ ?_).trans (le_max_right _ _)⟩
    <;> [left; right]
        /-
          case intro.refine_1.h
          α : Type u
          β : Type v
          inst✝¹ : TopologicalSpace α
          inst✝ : PseudoMetricSpace β
          f g : BoundedContinuousFunction α β
          C : Real
          hC : ∀ ⦃x : β⦄, Membership.mem (Union.union (Set.range ⇑f) (Set.range ⇑g)) x → …
          x : α
          ⊢ Membership.mem (Set.range ⇑f) (f x)
        -/
        /-
          🎉 no goals
        -/
    <;> apply mem_range_self
        /-
          🎉 no goals
        -/


/-- The pointwise distance is controlled by the distance between functions, by definition. -/
theorem dist_coe_le_dist (x : α) : dist (f x) (g x) ≤ dist f g :=
  le_csInf dist_set_exists fun _ hb => hb.2 x

/- This lemma will be needed in the proof of the metric space instance, but it will become
useless afterwards as it will be superseded by the general result that the distance is nonnegative
in metric spaces. -/

private theorem dist_nonneg' : 0 ≤ dist f g :=
  le_csInf dist_set_exists fun _ => And.left


/-- The distance between two functions is controlled by the supremum of the pointwise distances. -/
theorem dist_le (C0 : (0 : ℝ) ≤ C) : dist f g ≤ C ↔ ∀ x : α, dist (f x) (g x) ≤ C :=
  ⟨fun h x => le_trans (dist_coe_le_dist x) h, fun H => csInf_le ⟨0, fun _ => And.left⟩ ⟨C0, H⟩⟩


theorem dist_le_iff_of_nonempty [Nonempty α] : dist f g ≤ C ↔ ∀ x, dist (f x) (g x) ≤ C :=
  ⟨fun h x => le_trans (dist_coe_le_dist x) h,
    fun w => (dist_le (le_trans dist_nonneg (w (Nonempty.some ‹_›)))).mpr w⟩


theorem dist_lt_of_nonempty_compact [Nonempty α] [CompactSpace α]
    (w : ∀ x : α, dist (f x) (g x) < C) : dist f g < C := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    C : Real
    inst✝¹ : Nonempty α
    inst✝ : CompactSpace α
    w : ∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C
    ⊢ LT.lt (Dist.dist f g) C
  -/
  have c : Continuous fun x => dist (f x) (g x) := by continuity
  obtain ⟨x, -, le⟩ :=
    IsCompact.exists_isMaxOn isCompact_univ Set.univ_nonempty (Continuous.continuousOn c)
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    C : Real
    inst✝¹ : Nonempty α
    inst✝ : CompactSpace α
    w : ∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C
    c : Continuous fun x => Dist.dist (f x) (g x)
    x : α
    le : IsMaxOn (fun x => Dist.dist (f x) (g x)) Set.univ x
    ⊢ LT.lt (Dist.dist f g) C
  -/
  exact lt_of_le_of_lt (dist_le_iff_of_nonempty.mpr fun y => le trivial) (w x)
  /-
    🎉 no goals
  -/


theorem dist_lt_iff_of_compact [CompactSpace α] (C0 : (0 : ℝ) < C) :
    dist f g < C ↔ ∀ x : α, dist (f x) (g x) < C := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    C : Real
    inst✝ : CompactSpace α
    C0 : LT.lt 0 C
    ⊢ Iff (LT.lt (Dist.dist f g) C) (∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C)
  -/
  fconstructor
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : PseudoMetricSpace β
      f g : BoundedContinuousFunction α β
      C : Real
      inst✝ : CompactSpace α
      C0 : LT.lt 0 C
      ⊢ LT.lt (Dist.dist f g) C → ∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C
    -/
  · intro w x
    /-
      case mp
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : PseudoMetricSpace β
      f g : BoundedContinuousFunction α β
      C : Real
      inst✝ : CompactSpace α
      C0 : LT.lt 0 C
      w : LT.lt (Dist.dist f g) C
      x : α
      ⊢ LT.lt (Dist.dist (f x) (g x)) C
    -/
    exact lt_of_le_of_lt (dist_coe_le_dist x) w
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : PseudoMetricSpace β
      f g : BoundedContinuousFunction α β
      C : Real
      inst✝ : CompactSpace α
      C0 : LT.lt 0 C
      ⊢ (∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C) → LT.lt (Dist.dist f g) C
    -/
  · by_cases h : Nonempty α
      /-
        case pos
        α : Type u
        β : Type v
        inst✝² : TopologicalSpace α
        inst✝¹ : PseudoMetricSpace β
        f g : BoundedContinuousFunction α β
        C : Real
        inst✝ : CompactSpace α
        C0 : LT.lt 0 C
        h : Nonempty α
        ⊢ (∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C) → LT.lt (Dist.dist f g) C
      -/
    · exact dist_lt_of_nonempty_compact
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u
        β : Type v
        inst✝² : TopologicalSpace α
        inst✝¹ : PseudoMetricSpace β
        f g : BoundedContinuousFunction α β
        C : Real
        inst✝ : CompactSpace α
        C0 : LT.lt 0 C
        h : Not (Nonempty α)
        ⊢ (∀ (x : α), LT.lt (Dist.dist (f x) (g x)) C) → LT.lt (Dist.dist f g) C
      -/
    · rintro -
      /-
        case neg
        α : Type u
        β : Type v
        inst✝² : TopologicalSpace α
        inst✝¹ : PseudoMetricSpace β
        f g : BoundedContinuousFunction α β
        C : Real
        inst✝ : CompactSpace α
        C0 : LT.lt 0 C
        h : Not (Nonempty α)
        ⊢ LT.lt (Dist.dist f g) C
      -/
      convert C0
      /-
        case h.e'_3
        α : Type u
        β : Type v
        inst✝² : TopologicalSpace α
        inst✝¹ : PseudoMetricSpace β
        f g : BoundedContinuousFunction α β
        C : Real
        inst✝ : CompactSpace α
        C0 : LT.lt 0 C
        h : Not (Nonempty α)
        ⊢ Eq (Dist.dist f g) 0
      -/
      apply le_antisymm _ dist_nonneg'
      /-
        α : Type u
        β : Type v
        inst✝² : TopologicalSpace α
        inst✝¹ : PseudoMetricSpace β
        f g : BoundedContinuousFunction α β
        C : Real
        inst✝ : CompactSpace α
        C0 : LT.lt 0 C
        h : Not (Nonempty α)
        ⊢ LE.le (Dist.dist f g) 0
      -/
      rw [dist_eq]
      /-
        α : Type u
        β : Type v
        inst✝² : TopologicalSpace α
        inst✝¹ : PseudoMetricSpace β
        f g : BoundedContinuousFunction α β
        C : Real
        inst✝ : CompactSpace α
        C0 : LT.lt 0 C
        h : Not (Nonempty α)
        ⊢ LE.le (InfSet.sInf (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Dist.d …
      -/
      exact csInf_le ⟨0, fun C => And.left⟩ ⟨le_rfl, fun x => False.elim (h (Nonempty.intro x))⟩
      /-
        🎉 no goals
      -/


theorem dist_lt_iff_of_nonempty_compact [Nonempty α] [CompactSpace α] :
    dist f g < C ↔ ∀ x : α, dist (f x) (g x) < C :=
  ⟨fun w x => lt_of_le_of_lt (dist_coe_le_dist x) w, dist_lt_of_nonempty_compact⟩


/-- The type of bounded continuous functions, with the uniform distance, is a pseudometric space. -/
instance instPseudoMetricSpace : PseudoMetricSpace (α →ᵇ β) where
                                                             /-
                                                               F : Type u_1
                                                               α : Type u
                                                               β : Type v
                                                               γ : Type w
                                                               inst✝² : TopologicalSpace α
                                                               inst✝¹ : PseudoMetricSpace β
                                                               inst✝ : PseudoMetricSpace γ
                                                               f✝ g : BoundedContinuousFunction α β
                                                               x✝ : α
                                                               C : Real
                                                               f : BoundedContinuousFunction α β
                                                               x : α
                                                               ⊢ LE.le (Dist.dist (f x) (f x)) 0
                                                             -/
  dist_self f := le_antisymm ((dist_le le_rfl).2 fun x => by simp) dist_nonneg'
                                                             /-
                                                               🎉 no goals
                                                             -/
                      /-
                        F : Type u_1
                        α : Type u
                        β : Type v
                        γ : Type w
                        inst✝² : TopologicalSpace α
                        inst✝¹ : PseudoMetricSpace β
                        inst✝ : PseudoMetricSpace γ
                        f✝ g✝ : BoundedContinuousFunction α β
                        x : α
                        C : Real
                        f g : BoundedContinuousFunction α β
                        ⊢ Eq (Dist.dist f g) (Dist.dist g f)
                      -/
  dist_comm f g := by simp [dist_eq, dist_comm]
                      /-
                        🎉 no goals
                      -/
  dist_triangle _ _ _ := (dist_le (add_nonneg dist_nonneg' dist_nonneg')).2
    fun _ => le_trans (dist_triangle _ _ _) (add_le_add (dist_coe_le_dist _) (dist_coe_le_dist _))
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): added proof for `edist_dist`
                       /-
                         F : Type u_1
                         α : Type u
                         β : Type v
                         γ : Type w
                         inst✝² : TopologicalSpace α
                         inst✝¹ : PseudoMetricSpace β
                         inst✝ : PseudoMetricSpace γ
                         f g : BoundedContinuousFunction α β
                         x✝ : α
                         C : Real
                         x y : BoundedContinuousFunction α β
                         ⊢ Eq ((fun x y => ↑⟨Dist.dist x y, ⋯⟩) x y) (ENNReal.ofReal (Dist.dist x y))
                       -/
  edist_dist x y := by dsimp; congr; simp [dist_nonneg']
                                     /-
                                       🎉 no goals
                                     -/


/-- The type of bounded continuous functions, with the uniform distance, is a metric space. -/
instance instMetricSpace {β} [MetricSpace β] : MetricSpace (α →ᵇ β) where
  eq_of_dist_eq_zero hfg := by
    /-
      F : Type u_1
      α : Type u
      β✝ : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β✝
      inst✝¹ : PseudoMetricSpace γ
      f g : BoundedContinuousFunction α β✝
      x : α
      C : Real
      β : Type ?u.37788
      inst✝ : MetricSpace β
      x✝ y✝ : BoundedContinuousFunction α β
      hfg : Eq (Dist.dist x✝ y✝) 0
      ⊢ Eq x✝ y✝
    -/
    ext x
    /-
      case h
      F : Type u_1
      α : Type u
      β✝ : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β✝
      inst✝¹ : PseudoMetricSpace γ
      f g : BoundedContinuousFunction α β✝
      x✝¹ : α
      C : Real
      β : Type ?u.37788
      inst✝ : MetricSpace β
      x✝ y✝ : BoundedContinuousFunction α β
      hfg : Eq (Dist.dist x✝ y✝) 0
      x : α
      ⊢ Eq (x✝ x) (y✝ x)
    -/
    exact eq_of_dist_eq_zero (le_antisymm (hfg ▸ dist_coe_le_dist _) dist_nonneg)
    /-
      🎉 no goals
    -/


theorem nndist_eq : nndist f g = sInf { C | ∀ x : α, nndist (f x) (g x) ≤ C } :=
  Subtype.ext <| dist_eq.trans <| by
    /-
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : PseudoMetricSpace β
      f g : BoundedContinuousFunction α β
      ⊢ Eq (InfSet.sInf (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Dist.dist …
    -/
    rw [val_eq_coe, coe_sInf, coe_image]
    /-
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : PseudoMetricSpace β
      f g : BoundedContinuousFunction α β
      ⊢ Eq (InfSet.sInf (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Dist.dist …
    -/
    simp_rw [mem_setOf_eq, ← NNReal.coe_le_coe, coe_mk, exists_prop, coe_nndist]
    /-
      🎉 no goals
    -/


theorem nndist_set_exists : ∃ C, ∀ x : α, nndist (f x) (g x) ≤ C :=
  Subtype.exists.mpr <| dist_set_exists.imp fun _ ⟨ha, h⟩ => ⟨ha, h⟩


theorem nndist_coe_le_nndist (x : α) : nndist (f x) (g x) ≤ nndist f g :=
  dist_coe_le_dist x


/-- On an empty space, bounded continuous functions are at distance 0. -/
theorem dist_zero_of_empty [IsEmpty α] : dist f g = 0 := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    inst✝ : IsEmpty α
    ⊢ Eq (Dist.dist f g) 0
  -/
  rw [(ext isEmptyElim : f = g), dist_self]
  /-
    🎉 no goals
  -/


theorem dist_eq_iSup : dist f g = ⨆ x : α, dist (f x) (g x) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    ⊢ Eq (Dist.dist f g) (iSup fun x => Dist.dist (f x) (g x))
  -/
  cases isEmpty_or_nonempty α
    /-
      case inl
      α : Type u
      β : Type v
      inst✝¹ : TopologicalSpace α
      inst✝ : PseudoMetricSpace β
      f g : BoundedContinuousFunction α β
      h✝ : IsEmpty α
      ⊢ Eq (Dist.dist f g) (iSup fun x => Dist.dist (f x) (g x))
    -/
  · rw [iSup_of_empty', Real.sSup_empty, dist_zero_of_empty]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    h✝ : Nonempty α
    ⊢ Eq (Dist.dist f g) (iSup fun x => Dist.dist (f x) (g x))
  -/
  refine (dist_le_iff_of_nonempty.mpr <| le_ciSup ?_).antisymm (ciSup_le dist_coe_le_dist)
  /-
    case inr
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    f g : BoundedContinuousFunction α β
    h✝ : Nonempty α
    ⊢ BddAbove (Set.range fun x => Dist.dist (f x) (g x))
  -/
  exact dist_set_exists.imp fun C hC => forall_mem_range.2 hC.2
  /-
    🎉 no goals
  -/


theorem nndist_eq_iSup : nndist f g = ⨆ x : α, nndist (f x) (g x) :=
                                          /-
                                            α : Type u
                                            β : Type v
                                            inst✝¹ : TopologicalSpace α
                                            inst✝ : PseudoMetricSpace β
                                            f g : BoundedContinuousFunction α β
                                            ⊢ Eq (iSup fun x => Dist.dist (f x) (g x)) ↑(iSup fun x => NNDist.nndist (f x) …
                                          -/
  Subtype.ext <| dist_eq_iSup.trans <| by simp_rw [val_eq_coe, coe_iSup, coe_nndist]
                                          /-
                                            🎉 no goals
                                          -/


theorem tendsto_iff_tendstoUniformly {ι : Type*} {F : ι → α →ᵇ β} {f : α →ᵇ β} {l : Filter ι} :
    Tendsto F l (𝓝 f) ↔ TendstoUniformly (fun i => F i) f l :=
  Iff.intro
    (fun h =>
      tendstoUniformly_iff.2 fun ε ε0 =>
        (Metric.tendsto_nhds.mp h ε ε0).mp
          (Eventually.of_forall fun n hn x =>
            lt_of_le_of_lt (dist_coe_le_dist x) (dist_comm (F n) f ▸ hn)))
    fun h =>
    Metric.tendsto_nhds.mpr fun _ ε_pos =>
      (h _ (dist_mem_uniformity <| half_pos ε_pos)).mp
        (Eventually.of_forall fun n hn =>
          lt_of_le_of_lt
            ((dist_le (half_pos ε_pos).le).mpr fun x => dist_comm (f x) (F n x) ▸ le_of_lt (hn x))
            (half_lt_self ε_pos))


/-- The topology on `α →ᵇ β` is exactly the topology induced by the natural map to `α →ᵤ β`. -/
theorem isInducing_coeFn : IsInducing (UniformFun.ofFun ∘ (⇑) : (α →ᵇ β) → α →ᵤ β) := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    ⊢ Topology.IsInducing (Function.comp (⇑UniformFun.ofFun) DFunLike.coe)
  -/
  rw [isInducing_iff_nhds]
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    ⊢ ∀ (x : BoundedContinuousFunction α β), Eq (nhds x) (Filter.comap (Function.c …
  -/
  refine fun f => eq_of_forall_le_iff fun l => ?_
  rw [← tendsto_iff_comap, ← tendsto_id', tendsto_iff_tendstoUniformly,
    UniformFun.tendsto_iff_tendstoUniformly]
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : PseudoMetricSpace β
    f : BoundedContinuousFunction α β
    l : Filter (BoundedContinuousFunction α β)
    ⊢ Iff (TendstoUniformly (fun i => ⇑(id i)) (⇑f) l) (TendstoUniformly (Function …
  -/
  simp [comp_def]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-28")] alias inducing_coeFn := isInducing_coeFn

-- TODO: upgrade to `IsUniformEmbedding`

theorem isEmbedding_coeFn : IsEmbedding (UniformFun.ofFun ∘ (⇑) : (α →ᵇ β) → α →ᵤ β) :=
  ⟨isInducing_coeFn, fun _ _ h => ext fun x => congr_fun h x⟩


@[deprecated (since := "2024-10-26")]
alias embedding_coeFn := isEmbedding_coeFn


/-- Constant as a continuous bounded function. -/
@[simps! (config := .asFn)] -- Porting note: Changed `simps` to `simps!`
def const (b : β) : α →ᵇ β :=
                                  /-
                                    F : Type u_1
                                    α : Type u
                                    β : Type v
                                    γ : Type w
                                    inst✝² : TopologicalSpace α
                                    inst✝¹ : PseudoMetricSpace β
                                    inst✝ : PseudoMetricSpace γ
                                    f g : BoundedContinuousFunction α β
                                    x : α
                                    C : Real
                                    b : β
                                    ⊢ ∀ (x y : α), LE.le (Dist.dist ((ContinuousMap.const α b).toFun x) ((Continuo …
                                  -/
  ⟨ContinuousMap.const α b, 0, by simp⟩
                                  /-
                                    🎉 no goals
                                  -/


theorem const_apply' (a : α) (b : β) : (const α b : α → β) a = b := rfl


/-- If the target space is inhabited, so is the space of bounded continuous functions. -/
instance [Inhabited β] : Inhabited (α →ᵇ β) :=
  ⟨const α default⟩


theorem lipschitz_evalx (x : α) : LipschitzWith 1 fun f : α →ᵇ β => f x :=
  LipschitzWith.mk_one fun _ _ => dist_coe_le_dist x


theorem uniformContinuous_coe : @UniformContinuous (α →ᵇ β) (α → β) _ _ (⇑) :=
  uniformContinuous_pi.2 fun x => (lipschitz_evalx x).uniformContinuous


theorem continuous_coe : Continuous fun (f : α →ᵇ β) x => f x :=
  UniformContinuous.continuous uniformContinuous_coe


/-- When `x` is fixed, `(f : α →ᵇ β) ↦ f x` is continuous. -/
@[continuity]
theorem continuous_eval_const {x : α} : Continuous fun f : α →ᵇ β => f x :=
  (continuous_apply x).comp continuous_coe


/-- The evaluation map is continuous, as a joint function of `u` and `x`. -/
@[continuity]
theorem continuous_eval : Continuous fun p : (α →ᵇ β) × α => p.1 p.2 :=
  (continuous_prod_of_continuous_lipschitzWith _ 1 fun f => f.continuous) <| lipschitz_evalx


/-- Bounded continuous functions taking values in a complete space form a complete space. -/
instance instCompleteSpace [CompleteSpace β] : CompleteSpace (α →ᵇ β) :=
  complete_of_cauchySeq_tendsto fun (f : ℕ → α →ᵇ β) (hf : CauchySeq f) => by
    /- We have to show that `f n` converges to a bounded continuous function.
      For this, we prove pointwise convergence to define the limit, then check
      it is a continuous bounded function, and then check the norm convergence. -/
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : PseudoMetricSpace γ
      f✝ g : BoundedContinuousFunction α β
      x : α
      C : Real
      inst✝ : CompleteSpace β
      f : Nat → BoundedContinuousFunction α β
      hf : CauchySeq f
      ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
    -/
    rcases cauchySeq_iff_le_tendsto_0.1 hf with ⟨b, b0, b_bound, b_lim⟩
    /-
      case intro.intro.intro
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : PseudoMetricSpace γ
      f✝ g : BoundedContinuousFunction α β
      x : α
      C : Real
      inst✝ : CompleteSpace β
      f : Nat → BoundedContinuousFunction α β
      hf : CauchySeq f
      b : Nat → Real
      b0 : ∀ (n : Nat), LE.le 0 (b n)
      b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
      b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
      ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
    -/
    have f_bdd := fun x n m N hn hm => le_trans (dist_coe_le_dist x) (b_bound n m N hn hm)
    have fx_cau : ∀ x, CauchySeq fun n => f n x :=
      fun x => cauchySeq_iff_le_tendsto_0.2 ⟨b, b0, f_bdd x, b_lim⟩
    /-
      case intro.intro.intro
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : PseudoMetricSpace γ
      f✝ g : BoundedContinuousFunction α β
      x : α
      C : Real
      inst✝ : CompleteSpace β
      f : Nat → BoundedContinuousFunction α β
      hf : CauchySeq f
      b : Nat → Real
      b0 : ∀ (n : Nat), LE.le 0 (b n)
      b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
      b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
      f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
      fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
      ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
    -/
    choose F hF using fun x => cauchySeq_tendsto_of_complete (fx_cau x)
    /- `F : α → β`, `hF : ∀ (x : α), Tendsto (fun n ↦ ↑(f n) x) atTop (𝓝 (F x))`
      `F` is the desired limit function. Check that it is uniformly approximated by `f N`. -/
    have fF_bdd : ∀ x N, dist (f N x) (F x) ≤ b N :=
      fun x N => le_of_tendsto (tendsto_const_nhds.dist (hF x))
        (Filter.eventually_atTop.2 ⟨N, fun n hn => f_bdd x N n N (le_refl N) hn⟩)
    /-
      case intro.intro.intro
      F✝ : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      inst✝¹ : PseudoMetricSpace γ
      f✝ g : BoundedContinuousFunction α β
      x : α
      C : Real
      inst✝ : CompleteSpace β
      f : Nat → BoundedContinuousFunction α β
      hf : CauchySeq f
      b : Nat → Real
      b0 : ∀ (n : Nat), LE.le 0 (b n)
      b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
      b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
      f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
      fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
      F : α → β
      hF : ∀ (x : α), Filter.Tendsto (fun n => (f n) x) Filter.atTop (nhds (F x))
      fF_bdd : ∀ (x : α) (N : Nat), LE.le (Dist.dist ((f N) x) (F x)) (b N)
      ⊢ Exists fun a => Filter.Tendsto f Filter.atTop (nhds a)
    -/
    refine ⟨⟨⟨F, ?_⟩, ?_⟩, ?_⟩
    · -- Check that `F` is continuous, as a uniform limit of continuous functions
      have : TendstoUniformly (fun n x => f n x) F atTop := by
        refine Metric.tendstoUniformly_iff.2 fun ε ε0 => ?_
        refine ((tendsto_order.1 b_lim).2 ε ε0).mono fun n hn x => ?_
        rw [dist_comm]
        exact lt_of_le_of_lt (fF_bdd x n) hn
      /-
        case intro.intro.intro.refine_1
        F✝ : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : PseudoMetricSpace γ
        f✝ g : BoundedContinuousFunction α β
        x : α
        C : Real
        inst✝ : CompleteSpace β
        f : Nat → BoundedContinuousFunction α β
        hf : CauchySeq f
        b : Nat → Real
        b0 : ∀ (n : Nat), LE.le 0 (b n)
        b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
        b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
        f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
        fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
        F : α → β
        hF : ∀ (x : α), Filter.Tendsto (fun n => (f n) x) Filter.atTop (nhds (F x))
        fF_bdd : ∀ (x : α) (N : Nat), LE.le (Dist.dist ((f N) x) (F x)) (b N)
        this : TendstoUniformly (fun n x => (f n) x) F Filter.atTop
        ⊢ Continuous F
      -/
      exact this.continuous (Eventually.of_forall fun N => (f N).continuous)
      /-
        🎉 no goals
      -/
    · -- Check that `F` is bounded
      /-
        case intro.intro.intro.refine_2
        F✝ : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : PseudoMetricSpace γ
        f✝ g : BoundedContinuousFunction α β
        x : α
        C : Real
        inst✝ : CompleteSpace β
        f : Nat → BoundedContinuousFunction α β
        hf : CauchySeq f
        b : Nat → Real
        b0 : ∀ (n : Nat), LE.le 0 (b n)
        b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
        b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
        f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
        fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
        F : α → β
        hF : ∀ (x : α), Filter.Tendsto (fun n => (f n) x) Filter.atTop (nhds (F x))
        fF_bdd : ∀ (x : α) (N : Nat), LE.le (Dist.dist ((f N) x) (F x)) (b N)
        ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := F, continuous_toFu …
      -/
      rcases (f 0).bounded with ⟨C, hC⟩
      /-
        case intro.intro.intro.refine_2.intro
        F✝ : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : PseudoMetricSpace γ
        f✝ g : BoundedContinuousFunction α β
        x : α
        C✝ : Real
        inst✝ : CompleteSpace β
        f : Nat → BoundedContinuousFunction α β
        hf : CauchySeq f
        b : Nat → Real
        b0 : ∀ (n : Nat), LE.le 0 (b n)
        b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
        b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
        f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
        fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
        F : α → β
        hF : ∀ (x : α), Filter.Tendsto (fun n => (f n) x) Filter.atTop (nhds (F x))
        fF_bdd : ∀ (x : α) (N : Nat), LE.le (Dist.dist ((f N) x) (F x)) (b N)
        C : Real
        hC : ∀ (x y : α), LE.le (Dist.dist ((f 0) x) ((f 0) y)) C
        ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := F, continuous_toFu …
      -/
      refine ⟨C + (b 0 + b 0), fun x y => ?_⟩
      calc
        dist (F x) (F y) ≤ dist (f 0 x) (f 0 y) + (dist (f 0 x) (F x) + dist (f 0 y) (F y)) :=
          dist_triangle4_left _ _ _ _
        _ ≤ C + (b 0 + b 0) := by mono
    · -- Check that `F` is close to `f N` in distance terms
      /-
        case intro.intro.intro.refine_3
        F✝ : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : PseudoMetricSpace γ
        f✝ g : BoundedContinuousFunction α β
        x : α
        C : Real
        inst✝ : CompleteSpace β
        f : Nat → BoundedContinuousFunction α β
        hf : CauchySeq f
        b : Nat → Real
        b0 : ∀ (n : Nat), LE.le 0 (b n)
        b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
        b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
        f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
        fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
        F : α → β
        hF : ∀ (x : α), Filter.Tendsto (fun n => (f n) x) Filter.atTop (nhds (F x))
        fF_bdd : ∀ (x : α) (N : Nat), LE.le (Dist.dist ((f N) x) (F x)) (b N)
        ⊢ Filter.Tendsto f Filter.atTop (nhds { toFun := F, continuous_toFun := ⋯, map …
      -/
      refine tendsto_iff_dist_tendsto_zero.2 (squeeze_zero (fun _ => dist_nonneg) ?_ b_lim)
      /-
        case intro.intro.intro.refine_3
        F✝ : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : PseudoMetricSpace γ
        f✝ g : BoundedContinuousFunction α β
        x : α
        C : Real
        inst✝ : CompleteSpace β
        f : Nat → BoundedContinuousFunction α β
        hf : CauchySeq f
        b : Nat → Real
        b0 : ∀ (n : Nat), LE.le 0 (b n)
        b_bound : ∀ (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist (f n) (f m …
        b_lim : Filter.Tendsto b Filter.atTop (nhds 0)
        f_bdd : ∀ (x : α) (n m N : Nat), LE.le N n → LE.le N m → LE.le (Dist.dist ((f  …
        fx_cau : ∀ (x : α), CauchySeq fun n => (f n) x
        F : α → β
        hF : ∀ (x : α), Filter.Tendsto (fun n => (f n) x) Filter.atTop (nhds (F x))
        fF_bdd : ∀ (x : α) (N : Nat), LE.le (Dist.dist ((f N) x) (F x)) (b N)
        ⊢ ∀ (t : Nat), LE.le (Dist.dist (f t) { toFun := F, continuous_toFun := ⋯, map …
      -/
      exact fun N => (dist_le (b0 _)).2 fun x => fF_bdd x N
      /-
        🎉 no goals
      -/


/-- Composition of a bounded continuous function and a continuous function. -/
def compContinuous {δ : Type*} [TopologicalSpace δ] (f : α →ᵇ β) (g : C(δ, α)) : δ →ᵇ β where
  toContinuousMap := f.1.comp g
  map_bounded' := f.map_bounded'.imp fun _ hC _ _ => hC _ _


@[simp]
theorem coe_compContinuous {δ : Type*} [TopologicalSpace δ] (f : α →ᵇ β) (g : C(δ, α)) :
    ⇑(f.compContinuous g) = f ∘ g := rfl


@[simp]
theorem compContinuous_apply {δ : Type*} [TopologicalSpace δ] (f : α →ᵇ β) (g : C(δ, α)) (x : δ) :
    f.compContinuous g x = f (g x) := rfl


theorem lipschitz_compContinuous {δ : Type*} [TopologicalSpace δ] (g : C(δ, α)) :
    LipschitzWith 1 fun f : α →ᵇ β => f.compContinuous g :=
  LipschitzWith.mk_one fun _ _ => (dist_le dist_nonneg).2 fun x => dist_coe_le_dist (g x)


theorem continuous_compContinuous {δ : Type*} [TopologicalSpace δ] (g : C(δ, α)) :
    Continuous fun f : α →ᵇ β => f.compContinuous g :=
  (lipschitz_compContinuous g).continuous


/-- Restrict a bounded continuous function to a set. -/
def restrict (f : α →ᵇ β) (s : Set α) : s →ᵇ β :=
  f.compContinuous <| (ContinuousMap.id _).restrict s


@[simp]
theorem coe_restrict (f : α →ᵇ β) (s : Set α) : ⇑(f.restrict s) = f ∘ (↑) := rfl


@[simp]
theorem restrict_apply (f : α →ᵇ β) (s : Set α) (x : s) : f.restrict s x = f x := rfl


/-- Composition (in the target) of a bounded continuous function with a Lipschitz map again
gives a bounded continuous function. -/
def comp (G : β → γ) {C : ℝ≥0} (H : LipschitzWith C G) (f : α →ᵇ β) : α →ᵇ γ :=
  ⟨⟨fun x => G (f x), H.continuous.comp f.continuous⟩,
    let ⟨D, hD⟩ := f.bounded
    ⟨max C 0 * D, fun x y =>
      calc
        dist (G (f x)) (G (f y)) ≤ C * dist (f x) (f y) := H.dist_le_mul _ _
                                             /-
                                               F : Type u_1
                                               α : Type u
                                               β : Type v
                                               γ : Type w
                                               inst✝² : TopologicalSpace α
                                               inst✝¹ : PseudoMetricSpace β
                                               inst✝ : PseudoMetricSpace γ
                                               f✝ g : BoundedContinuousFunction α β
                                               x✝ : α
                                               C✝ : Real
                                               G : β → γ
                                               C : NNReal
                                               H : LipschitzWith C G
                                               f : BoundedContinuousFunction α β
                                               D : Real
                                               hD : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) D
                                               x y : α
                                               ⊢ LE.le (HMul.hMul (↑C) (Dist.dist (f x) (f y))) (HMul.hMul (↑(Max.max C 0)) ( …
                                             -/
        _ ≤ max C 0 * dist (f x) (f y) := by gcongr; apply le_max_left
                                                     /-
                                                       🎉 no goals
                                                     -/
                              /-
                                F : Type u_1
                                α : Type u
                                β : Type v
                                γ : Type w
                                inst✝² : TopologicalSpace α
                                inst✝¹ : PseudoMetricSpace β
                                inst✝ : PseudoMetricSpace γ
                                f✝ g : BoundedContinuousFunction α β
                                x✝ : α
                                C✝ : Real
                                G : β → γ
                                C : NNReal
                                H : LipschitzWith C G
                                f : BoundedContinuousFunction α β
                                D : Real
                                hD : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) D
                                x y : α
                                ⊢ LE.le (HMul.hMul (↑(Max.max C 0)) (Dist.dist (f x) (f y))) (HMul.hMul (↑(Max …
                              -/
        _ ≤ max C 0 * D := by gcongr; apply hD
                                      /-
                                        🎉 no goals
                                      -/
        ⟩⟩


/-- The composition operator (in the target) with a Lipschitz map is Lipschitz. -/
theorem lipschitz_comp {G : β → γ} {C : ℝ≥0} (H : LipschitzWith C G) :
    LipschitzWith C (comp G H : (α →ᵇ β) → α →ᵇ γ) :=
  LipschitzWith.of_dist_le_mul fun f g =>
    (dist_le (mul_nonneg C.2 dist_nonneg)).2 fun x =>
      calc
        dist (G (f x)) (G (g x)) ≤ C * dist (f x) (g x) := H.dist_le_mul _ _
                               /-
                                 α : Type u
                                 β : Type v
                                 γ : Type w
                                 inst✝² : TopologicalSpace α
                                 inst✝¹ : PseudoMetricSpace β
                                 inst✝ : PseudoMetricSpace γ
                                 G : β → γ
                                 C : NNReal
                                 H : LipschitzWith C G
                                 f g : BoundedContinuousFunction α β
                                 x : α
                                 ⊢ LE.le (HMul.hMul (↑C) (Dist.dist (f x) (g x))) (HMul.hMul (↑C) (Dist.dist f  …
                               -/
        _ ≤ C * dist f g := by gcongr; apply dist_coe_le_dist
                                       /-
                                         🎉 no goals
                                       -/


/-- The composition operator (in the target) with a Lipschitz map is uniformly continuous. -/
theorem uniformContinuous_comp {G : β → γ} {C : ℝ≥0} (H : LipschitzWith C G) :
    UniformContinuous (comp G H : (α →ᵇ β) → α →ᵇ γ) :=
  (lipschitz_comp H).uniformContinuous


/-- The composition operator (in the target) with a Lipschitz map is continuous. -/
theorem continuous_comp {G : β → γ} {C : ℝ≥0} (H : LipschitzWith C G) :
    Continuous (comp G H : (α →ᵇ β) → α →ᵇ γ) :=
  (lipschitz_comp H).continuous


/-- Restriction (in the target) of a bounded continuous function taking values in a subset. -/
def codRestrict (s : Set β) (f : α →ᵇ β) (H : ∀ x, f x ∈ s) : α →ᵇ s :=
  ⟨⟨s.codRestrict f H, f.continuous.subtype_mk _⟩, f.bounded⟩


/-- A version of `Function.extend` for bounded continuous maps. We assume that the domain has
discrete topology, so we only need to verify boundedness. -/
nonrec def extend (f : α ↪ δ) (g : α →ᵇ β) (h : δ →ᵇ β) : δ →ᵇ β where
  toFun := extend f g h
  continuous_toFun := continuous_of_discreteTopology
  map_bounded' := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : PseudoMetricSpace β
      inst✝² : PseudoMetricSpace γ
      f✝ g✝ : BoundedContinuousFunction α β
      x : α
      C : Real
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g : BoundedContinuousFunction α β
      h : BoundedContinuousFunction δ β
      ⊢ Exists fun C => ∀ (x y : δ), LE.le (Dist.dist ({ toFun := Function.extend ⇑f …
    -/
    rw [← isBounded_range_iff, range_extend f.injective]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : PseudoMetricSpace β
      inst✝² : PseudoMetricSpace γ
      f✝ g✝ : BoundedContinuousFunction α β
      x : α
      C : Real
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g : BoundedContinuousFunction α β
      h : BoundedContinuousFunction δ β
      ⊢ Bornology.IsBounded (Union.union (Set.range ⇑g) (Set.image (⇑h) (HasCompl.co …
    -/
    exact g.isBounded_range.union (h.isBounded_image _)
    /-
      🎉 no goals
    -/


@[simp]
theorem extend_apply (f : α ↪ δ) (g : α →ᵇ β) (h : δ →ᵇ β) (x : α) : extend f g h (f x) = g x :=
  f.injective.extend_apply _ _ _


@[simp]
nonrec theorem extend_comp (f : α ↪ δ) (g : α →ᵇ β) (h : δ →ᵇ β) : extend f g h ∘ f = g :=
  extend_comp f.injective _ _


nonrec theorem extend_apply' {f : α ↪ δ} {x : δ} (hx : x ∉ range f) (g : α →ᵇ β) (h : δ →ᵇ β) :
    extend f g h x = h x :=
  extend_apply' _ _ _ hx


theorem extend_of_empty [IsEmpty α] (f : α ↪ δ) (g : α →ᵇ β) (h : δ →ᵇ β) : extend f g h = h :=
  DFunLike.coe_injective <| Function.extend_of_isEmpty f g h


@[simp]
theorem dist_extend_extend (f : α ↪ δ) (g₁ g₂ : α →ᵇ β) (h₁ h₂ : δ →ᵇ β) :
    dist (g₁.extend f h₁) (g₂.extend f h₂) =
      max (dist g₁ g₂) (dist (h₁.restrict (range f)ᶜ) (h₂.restrict (range f)ᶜ)) := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : PseudoMetricSpace β
    δ : Type u_2
    inst✝¹ : TopologicalSpace δ
    inst✝ : DiscreteTopology δ
    f : Function.Embedding α δ
    g₁ g₂ : BoundedContinuousFunction α β
    h₁ h₂ : BoundedContinuousFunction δ β
    ⊢ Eq (Dist.dist (BoundedContinuousFunction.extend f g₁ h₁) (BoundedContinuousF …
  -/
  refine le_antisymm ((dist_le <| le_max_iff.2 <| Or.inl dist_nonneg).2 fun x => ?_) (max_le ?_ ?_)
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g₁ g₂ : BoundedContinuousFunction α β
      h₁ h₂ : BoundedContinuousFunction δ β
      x : δ
      ⊢ LE.le (Dist.dist ((BoundedContinuousFunction.extend f g₁ h₁) x) ((BoundedCon …
    -/
  · rcases em (∃ y, f y = x) with (⟨x, rfl⟩ | hx)
      /-
        case refine_1.inl.intro
        α : Type u
        β : Type v
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        δ : Type u_2
        inst✝¹ : TopologicalSpace δ
        inst✝ : DiscreteTopology δ
        f : Function.Embedding α δ
        g₁ g₂ : BoundedContinuousFunction α β
        h₁ h₂ : BoundedContinuousFunction δ β
        x : α
        ⊢ LE.le (Dist.dist ((BoundedContinuousFunction.extend f g₁ h₁) (f x)) ((Bounde …
      -/
    · simp only [extend_apply]
      /-
        case refine_1.inl.intro
        α : Type u
        β : Type v
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        δ : Type u_2
        inst✝¹ : TopologicalSpace δ
        inst✝ : DiscreteTopology δ
        f : Function.Embedding α δ
        g₁ g₂ : BoundedContinuousFunction α β
        h₁ h₂ : BoundedContinuousFunction δ β
        x : α
        ⊢ LE.le (Dist.dist (g₁ x) (g₂ x)) (Max.max (Dist.dist g₁ g₂) (Dist.dist (h₁.re …
      -/
      exact (dist_coe_le_dist x).trans (le_max_left _ _)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.inr
        α : Type u
        β : Type v
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        δ : Type u_2
        inst✝¹ : TopologicalSpace δ
        inst✝ : DiscreteTopology δ
        f : Function.Embedding α δ
        g₁ g₂ : BoundedContinuousFunction α β
        h₁ h₂ : BoundedContinuousFunction δ β
        x : δ
        hx : Not (Exists fun y => Eq (f y) x)
        ⊢ LE.le (Dist.dist ((BoundedContinuousFunction.extend f g₁ h₁) x) ((BoundedCon …
      -/
    · simp only [extend_apply' hx]
      /-
        case refine_1.inr
        α : Type u
        β : Type v
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        δ : Type u_2
        inst✝¹ : TopologicalSpace δ
        inst✝ : DiscreteTopology δ
        f : Function.Embedding α δ
        g₁ g₂ : BoundedContinuousFunction α β
        h₁ h₂ : BoundedContinuousFunction δ β
        x : δ
        hx : Not (Exists fun y => Eq (f y) x)
        ⊢ LE.le (Dist.dist (h₁ x) (h₂ x)) (Max.max (Dist.dist g₁ g₂) (Dist.dist (h₁.re …
      -/
      lift x to ((range f)ᶜ : Set δ) using hx
      calc
        dist (h₁ x) (h₂ x) = dist (h₁.restrict (range f)ᶜ x) (h₂.restrict (range f)ᶜ x) := rfl
        _ ≤ dist (h₁.restrict (range f)ᶜ) (h₂.restrict (range f)ᶜ) := dist_coe_le_dist x
        _ ≤ _ := le_max_right _ _
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g₁ g₂ : BoundedContinuousFunction α β
      h₁ h₂ : BoundedContinuousFunction δ β
      ⊢ LE.le (Dist.dist g₁ g₂) (Dist.dist (BoundedContinuousFunction.extend f g₁ h₁ …
    -/
  · refine (dist_le dist_nonneg).2 fun x => ?_
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g₁ g₂ : BoundedContinuousFunction α β
      h₁ h₂ : BoundedContinuousFunction δ β
      x : α
      ⊢ LE.le (Dist.dist (g₁ x) (g₂ x)) (Dist.dist (BoundedContinuousFunction.extend …
    -/
    rw [← extend_apply f g₁ h₁, ← extend_apply f g₂ h₂]
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g₁ g₂ : BoundedContinuousFunction α β
      h₁ h₂ : BoundedContinuousFunction δ β
      x : α
      ⊢ LE.le (Dist.dist ((BoundedContinuousFunction.extend f g₁ h₁) (f x)) ((Bounde …
    -/
    exact dist_coe_le_dist _
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u
      β : Type v
      inst✝³ : TopologicalSpace α
      inst✝² : PseudoMetricSpace β
      δ : Type u_2
      inst✝¹ : TopologicalSpace δ
      inst✝ : DiscreteTopology δ
      f : Function.Embedding α δ
      g₁ g₂ : BoundedContinuousFunction α β
      h₁ h₂ : BoundedContinuousFunction δ β
      ⊢ LE.le (Dist.dist (h₁.restrict (HasCompl.compl (Set.range ⇑f))) (h₂.restrict  …
    -/
  · refine (dist_le dist_nonneg).2 fun x => ?_
    calc
      dist (h₁ x) (h₂ x) = dist (extend f g₁ h₁ x) (extend f g₂ h₂ x) := by
        rw [extend_apply' x.coe_prop, extend_apply' x.coe_prop]
      _ ≤ _ := dist_coe_le_dist _


theorem isometry_extend (f : α ↪ δ) (h : δ →ᵇ β) : Isometry fun g : α →ᵇ β => extend f g h :=
                                      /-
                                        α : Type u
                                        β : Type v
                                        inst✝³ : TopologicalSpace α
                                        inst✝² : PseudoMetricSpace β
                                        δ : Type u_2
                                        inst✝¹ : TopologicalSpace δ
                                        inst✝ : DiscreteTopology δ
                                        f : Function.Embedding α δ
                                        h : BoundedContinuousFunction δ β
                                        g₁ g₂ : BoundedContinuousFunction α β
                                        ⊢ Eq (Dist.dist (BoundedContinuousFunction.extend f g₁ h) (BoundedContinuousFu …
                                      -/
  Isometry.of_dist_eq fun g₁ g₂ => by simp [dist_nonneg]
                                      /-
                                        🎉 no goals
                                      -/


/-- The indicator function of a clopen set, as a bounded continuous function. -/
@[simps]
noncomputable def indicator (s : Set α) (hs : IsClopen s) : BoundedContinuousFunction α ℝ where
  toFun := s.indicator 1
                                               /-
                                                 F : Type u_1
                                                 α : Type u
                                                 β : Type v
                                                 γ : Type w
                                                 inst✝² : TopologicalSpace α
                                                 inst✝¹ : PseudoMetricSpace β
                                                 inst✝ : PseudoMetricSpace γ
                                                 f g : BoundedContinuousFunction α β
                                                 x : α
                                                 C : Real
                                                 s : Set α
                                                 hs : IsClopen s
                                                 ⊢ ∀ (a : α), Membership.mem (frontier s) a → Eq (1 a) 0
                                               -/
  continuous_toFun := continuous_indicator (by simp [hs]) <| continuous_const.continuousOn
                                               /-
                                                 🎉 no goals
                                               -/
                                   /-
                                     F : Type u_1
                                     α : Type u
                                     β : Type v
                                     γ : Type w
                                     inst✝² : TopologicalSpace α
                                     inst✝¹ : PseudoMetricSpace β
                                     inst✝ : PseudoMetricSpace γ
                                     f g : BoundedContinuousFunction α β
                                     x✝ : α
                                     C : Real
                                     s : Set α
                                     hs : IsClopen s
                                     x y : α
                                     ⊢ LE.le (Dist.dist ({ toFun := s.indicator 1, continuous_toFun := ⋯ }.toFun x) …
                                   -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  map_bounded' := ⟨1, fun x y ↦ by by_cases hx : x ∈ s <;> by_cases hy : y ∈ s <;> simp [hx, hy]⟩
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


/-- First version, with pointwise equicontinuity and range in a compact space. -/
theorem arzela_ascoli₁ [CompactSpace β] (A : Set (α →ᵇ β)) (closed : IsClosed A)
    (H : Equicontinuous ((↑) : A → α → β)) : IsCompact A := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : Equicontinuous fun x => ⇑↑x
    ⊢ IsCompact A
  -/
  simp_rw [Equicontinuous, Metric.equicontinuousAt_iff_pair] at H
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ⊢ IsCompact A
  -/
  refine isCompact_of_totallyBounded_isClosed ?_ closed
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ⊢ TotallyBounded A
  -/
  refine totallyBounded_of_finite_discretization fun ε ε0 => ?_
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  rcases exists_between ε0 with ⟨ε₁, ε₁0, εε₁⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    εε₁ : LT.lt ε₁ ε
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  let ε₂ := ε₁ / 2 / 2
  /- We have to find a finite discretization of `u`, i.e., finite information
    that is sufficient to reconstruct `u` up to `ε`. This information will be
    provided by the values of `u` on a sufficiently dense set `tα`,
    slightly translated to fit in a finite `ε₂`-dense set `tβ` in the image. Such
    sets exist by compactness of the source and range. Then, to check that these
    data determine the function up to `ε`, one uses the control on the modulus of
    continuity to extend the closeness on tα to closeness everywhere. -/
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    εε₁ : LT.lt ε₁ ε
    ε₂ : Real := HDiv.hDiv (HDiv.hDiv ε₁ 2) 2
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  have ε₂0 : ε₂ > 0 := half_pos (half_pos ε₁0)
  have : ∀ x : α, ∃ U, x ∈ U ∧ IsOpen U ∧
      ∀ y ∈ U, ∀ z ∈ U, ∀ {f : α →ᵇ β}, f ∈ A → dist (f y) (f z) < ε₂ := fun x =>
    let ⟨U, nhdsU, hU⟩ := H x _ ε₂0
    let ⟨V, VU, openV, xV⟩ := _root_.mem_nhds_iff.1 nhdsU
    ⟨V, xV, openV, fun y hy z hz f hf => hU y (VU hy) z (VU hz) ⟨f, hf⟩⟩
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    εε₁ : LT.lt ε₁ ε
    ε₂ : Real := HDiv.hDiv (HDiv.hDiv ε₁ 2) 2
    ε₂0 : GT.gt ε₂ 0
    this : ∀ (x : α), Exists fun U => And (Membership.mem U x) (And (IsOpen U) (∀  …
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  choose U hU using this
  /- For all `x`, the set `hU x` is an open set containing `x` on which the elements of `A`
    fluctuate by at most `ε₂`.
    We extract finitely many of these sets that cover the whole space, by compactness. -/
  obtain ⟨tα : Set α, _, hfin, htα : univ ⊆ ⋃ x ∈ tα, U x⟩ :=
    isCompact_univ.elim_finite_subcover_image (fun x _ => (hU x).2.1) fun x _ =>
      mem_biUnion (mem_univ _) (hU x).1
  /-
    case intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    εε₁ : LT.lt ε₁ ε
    ε₂ : Real := HDiv.hDiv (HDiv.hDiv ε₁ 2) 2
    ε₂0 : GT.gt ε₂ 0
    U : α → Set α
    hU : ∀ (x : α), And (Membership.mem (U x) x) (And (IsOpen (U x)) (∀ (y : α), M …
    tα : Set α
    left✝ : HasSubset.Subset tα Set.univ
    hfin : tα.Finite
    htα : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U x)
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  rcases hfin.nonempty_fintype with ⟨_⟩
  obtain ⟨tβ : Set β, _, hfin, htβ : univ ⊆ ⋃y ∈ tβ, ball y ε₂⟩ :=
    @finite_cover_balls_of_compact β _ _ isCompact_univ _ ε₂0
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    εε₁ : LT.lt ε₁ ε
    ε₂ : Real := HDiv.hDiv (HDiv.hDiv ε₁ 2) 2
    ε₂0 : GT.gt ε₂ 0
    U : α → Set α
    hU : ∀ (x : α), And (Membership.mem (U x) x) (And (IsOpen (U x)) (∀ (y : α), M …
    tα : Set α
    left✝¹ : HasSubset.Subset tα Set.univ
    hfin✝ : tα.Finite
    htα : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U x)
    val✝ : Fintype ↑tα
    tβ : Set β
    left✝ : HasSubset.Subset tβ Set.univ
    hfin : tβ.Finite
    htβ : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => Metri …
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  rcases hfin.nonempty_fintype with ⟨_⟩
  -- Associate to every point `y` in the space a nearby point `F y` in `tβ`
  /-
    case intro.intro.intro.intro.intro.intro.intro.intro.intro.intro
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : CompactSpace α
    inst✝¹ : PseudoMetricSpace β
    inst✝ : CompactSpace β
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    H : ∀ (x₀ : α) (ε : Real), GT.gt ε 0 → Exists fun U => And (Membership.mem (nh …
    ε : Real
    ε0 : GT.gt ε 0
    ε₁ : Real
    ε₁0 : LT.lt 0 ε₁
    εε₁ : LT.lt ε₁ ε
    ε₂ : Real := HDiv.hDiv (HDiv.hDiv ε₁ 2) 2
    ε₂0 : GT.gt ε₂ 0
    U : α → Set α
    hU : ∀ (x : α), And (Membership.mem (U x) x) (And (IsOpen (U x)) (∀ (y : α), M …
    tα : Set α
    left✝¹ : HasSubset.Subset tα Set.univ
    hfin✝ : tα.Finite
    htα : HasSubset.Subset Set.univ (Set.iUnion fun x => Set.iUnion fun h => U x)
    val✝¹ : Fintype ↑tα
    tβ : Set β
    left✝ : HasSubset.Subset tβ Set.univ
    hfin : tβ.Finite
    htβ : HasSubset.Subset Set.univ (Set.iUnion fun y => Set.iUnion fun h => Metri …
    val✝ : Fintype ↑tβ
    ⊢ Exists fun β_1 => Exists fun x => Exists fun F => ∀ (x y : ↑A), Eq (F x) (F  …
  -/
  choose F hF using fun y => show ∃ z ∈ tβ, dist y z < ε₂ by simpa using htβ (mem_univ y)
  -- `F : β → β`, `hF : ∀ (y : β), F y ∈ tβ ∧ dist y (F y) < ε₂`
  /- Associate to every function a discrete approximation, mapping each point in `tα`
    to a point in `tβ` close to its true image by the function. -/
  classical
  refine ⟨tα → tβ, by infer_instance, fun f a => ⟨F (f.1 a), (hF (f.1 a)).1⟩, ?_⟩
  rintro ⟨f, hf⟩ ⟨g, hg⟩ f_eq_g
  -- If two functions have the same approximation, then they are within distance `ε`
  refine lt_of_le_of_lt ((dist_le <| le_of_lt ε₁0).2 fun x => ?_) εε₁
  obtain ⟨x', x'tα, hx'⟩ := mem_iUnion₂.1 (htα (mem_univ x))
  calc
    dist (f x) (g x) ≤ dist (f x) (f x') + dist (g x) (g x') + dist (f x') (g x') :=
      dist_triangle4_right _ _ _ _
    _ ≤ ε₂ + ε₂ + ε₁ / 2 := by
      refine le_of_lt (add_lt_add (add_lt_add ?_ ?_) ?_)
      · exact (hU x').2.2 _ hx' _ (hU x').1 hf
      · exact (hU x').2.2 _ hx' _ (hU x').1 hg
      · have F_f_g : F (f x') = F (g x') :=
          (congr_arg (fun f : tα → tβ => (f ⟨x', x'tα⟩ : β)) f_eq_g : _)
        calc
          dist (f x') (g x') ≤ dist (f x') (F (f x')) + dist (g x') (F (f x')) :=
            dist_triangle_right _ _ _
          _ = dist (f x') (F (f x')) + dist (g x') (F (g x')) := by rw [F_f_g]
          _ < ε₂ + ε₂ := (add_lt_add (hF (f x')).2 (hF (g x')).2)
          _ = ε₁ / 2 := add_halves _
    _ = ε₁ := by rw [add_halves, add_halves]


/-- Second version, with pointwise equicontinuity and range in a compact subset. -/
theorem arzela_ascoli₂ (s : Set β) (hs : IsCompact s) (A : Set (α →ᵇ β)) (closed : IsClosed A)
    (in_s : ∀ (f : α →ᵇ β) (x : α), f ∈ A → f x ∈ s) (H : Equicontinuous ((↑) : A → α → β)) :
    IsCompact A := by
  /- This version is deduced from the previous one by restricting to the compact type in the target,
  using compactness there and then lifting everything to the original space. -/
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : CompactSpace α
    inst✝ : PseudoMetricSpace β
    s : Set β
    hs : IsCompact s
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
    H : Equicontinuous fun x => ⇑↑x
    ⊢ IsCompact A
  -/
  have M : LipschitzWith 1 Subtype.val := LipschitzWith.subtype_val s
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : CompactSpace α
    inst✝ : PseudoMetricSpace β
    s : Set β
    hs : IsCompact s
    A : Set (BoundedContinuousFunction α β)
    closed : IsClosed A
    in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
    H : Equicontinuous fun x => ⇑↑x
    M : LipschitzWith 1 Subtype.val
    ⊢ IsCompact A
  -/
  let F : (α →ᵇ s) → α →ᵇ β := comp (↑) M
  refine IsCompact.of_isClosed_subset ((?_ : IsCompact (F ⁻¹' A)).image (continuous_comp M)) closed
      fun f hf => ?_
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      ⊢ IsCompact (Set.preimage F A)
    -/
  · haveI : CompactSpace s := isCompact_iff_compactSpace.1 hs
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      this : CompactSpace ↑s
      ⊢ IsCompact (Set.preimage F A)
    -/
    refine arzela_ascoli₁ _ (continuous_iff_isClosed.1 (continuous_comp M) _ closed) ?_
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      this : CompactSpace ↑s
      ⊢ Equicontinuous fun x => ⇑↑x
    -/
    rw [isUniformEmbedding_subtype_val.isUniformInducing.equicontinuous_iff]
    /-
      case refine_1
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      this : CompactSpace ↑s
      ⊢ Equicontinuous (Function.comp (fun x => Function.comp Subtype.val x) fun x = …
    -/
    exact H.comp (A.restrictPreimage F)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      f : BoundedContinuousFunction α β
      hf : Membership.mem A f
      ⊢ Membership.mem (Set.image (BoundedContinuousFunction.comp Subtype.val M) (Se …
    -/
  · let g := codRestrict s f fun x => in_s f x hf
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      f : BoundedContinuousFunction α β
      hf : Membership.mem A f
      g : BoundedContinuousFunction α ↑s := BoundedContinuousFunction.codRestrict s  …
      ⊢ Membership.mem (Set.image (BoundedContinuousFunction.comp Subtype.val M) (Se …
    -/
    rw [show f = F g by ext; rfl] at hf ⊢
    /-
      case refine_2
      α : Type u
      β : Type v
      inst✝² : TopologicalSpace α
      inst✝¹ : CompactSpace α
      inst✝ : PseudoMetricSpace β
      s : Set β
      hs : IsCompact s
      A : Set (BoundedContinuousFunction α β)
      closed : IsClosed A
      in_s : ∀ (f : BoundedContinuousFunction α β) (x : α), Membership.mem A f → Mem …
      H : Equicontinuous fun x => ⇑↑x
      M : LipschitzWith 1 Subtype.val
      F : BoundedContinuousFunction α ↑s → BoundedContinuousFunction α β := BoundedC …
      f : BoundedContinuousFunction α β
      hf✝ : Membership.mem A f
      g : BoundedContinuousFunction α ↑s := BoundedContinuousFunction.codRestrict s  …
      hf : Membership.mem A (F g)
      ⊢ Membership.mem (Set.image (BoundedContinuousFunction.comp Subtype.val M) (Se …
    -/
    exact ⟨g, hf, rfl⟩
    /-
      🎉 no goals
    -/


/-- Third (main) version, with pointwise equicontinuity and range in a compact subset, but
without closedness. The closure is then compact. -/
theorem arzela_ascoli [T2Space β] (s : Set β) (hs : IsCompact s) (A : Set (α →ᵇ β))
    (in_s : ∀ (f : α →ᵇ β) (x : α), f ∈ A → f x ∈ s) (H : Equicontinuous ((↑) : A → α → β)) :
    IsCompact (closure A) :=
  /- This version is deduced from the previous one by checking that the closure of `A`, in
  addition to being closed, still satisfies the properties of compact range and equicontinuity. -/
  arzela_ascoli₂ s hs (closure A) isClosed_closure
    (fun _ x hf =>
      (mem_of_closed' hs.isClosed).2 fun ε ε0 =>
        let ⟨g, gA, dist_fg⟩ := Metric.mem_closure_iff.1 hf ε ε0
        ⟨g x, in_s g x gA, lt_of_le_of_lt (dist_coe_le_dist _) dist_fg⟩)
    (H.closure' continuous_coe)


@[to_additive] instance instOne : One (α →ᵇ β) := ⟨const α 1⟩


@[to_additive (attr := simp)]
theorem coe_one : ((1 : α →ᵇ β) : α → β) = 1 := rfl


@[to_additive (attr := simp)]
theorem mkOfCompact_one [CompactSpace α] : mkOfCompact (1 : C(α, β)) = 1 := rfl


@[to_additive]
theorem forall_coe_one_iff_one (f : α →ᵇ β) : (∀ x, f x = 1) ↔ f = 1 :=
  (@DFunLike.ext_iff _ _ _ _ f 1).symm


@[to_additive (attr := simp)]
theorem one_compContinuous [TopologicalSpace γ] (f : C(γ, α)) : (1 : α →ᵇ β).compContinuous f = 1 :=
  rfl


/-- The pointwise sum of two bounded continuous functions is again bounded continuous. -/
instance instAdd : Add (α →ᵇ β) where
  add f g :=
    { toFun := fun x ↦ f x + g x
      continuous_toFun := f.continuous.add g.continuous
      map_bounded' := add_bounded_of_bounded_of_bounded (map_bounded f) (map_bounded g) }


@[simp]
theorem coe_add : ⇑(f + g) = f + g := rfl


theorem add_apply : (f + g) x = f x + g x := rfl


@[simp]
theorem mkOfCompact_add [CompactSpace α] (f g : C(α, β)) :
    mkOfCompact (f + g) = mkOfCompact f + mkOfCompact g := rfl


theorem add_compContinuous [TopologicalSpace γ] (h : C(γ, α)) :
    (g + f).compContinuous h = g.compContinuous h + f.compContinuous h := rfl


@[simp]
theorem coe_nsmulRec : ∀ n, ⇑(nsmulRec n f) = n • ⇑f
            /-
              α : Type u
              β : Type v
              inst✝⁴ : TopologicalSpace α
              inst✝³ : PseudoMetricSpace β
              inst✝² : AddMonoid β
              inst✝¹ : BoundedAdd β
              inst✝ : ContinuousAdd β
              f : BoundedContinuousFunction α β
              ⊢ Eq (⇑(nsmulRec 0 f)) (HSMul.hSMul 0 ⇑f)
            -/
  | 0 => by rw [nsmulRec, zero_smul, coe_zero]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u
                  β : Type v
                  inst✝⁴ : TopologicalSpace α
                  inst✝³ : PseudoMetricSpace β
                  inst✝² : AddMonoid β
                  inst✝¹ : BoundedAdd β
                  inst✝ : ContinuousAdd β
                  f : BoundedContinuousFunction α β
                  n : Nat
                  ⊢ Eq (⇑(nsmulRec (HAdd.hAdd n 1) f)) (HSMul.hSMul (HAdd.hAdd n 1) ⇑f)
                -/
  | n + 1 => by rw [nsmulRec, succ_nsmul, coe_add, coe_nsmulRec n]
                /-
                  🎉 no goals
                -/


instance instSMulNat : SMul ℕ (α →ᵇ β) where
  smul n f :=
    { toContinuousMap := n • f.toContinuousMap
                         /-
                           F : Type u_1
                           α : Type u
                           β : Type v
                           γ : Type w
                           inst✝⁴ : TopologicalSpace α
                           inst✝³ : PseudoMetricSpace β
                           inst✝² : AddMonoid β
                           inst✝¹ : BoundedAdd β
                           inst✝ : ContinuousAdd β
                           f✝ g : BoundedContinuousFunction α β
                           x : α
                           C : Real
                           n : Nat
                           f : BoundedContinuousFunction α β
                           ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ((HSMul.hSMul n f.toContinuous …
                         -/
      map_bounded' := by simpa [coe_nsmulRec] using (nsmulRec n f).map_bounded' }
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem coe_nsmul (r : ℕ) (f : α →ᵇ β) : ⇑(r • f) = r • ⇑f := rfl


@[simp]
theorem nsmul_apply (r : ℕ) (f : α →ᵇ β) (v : α) : (r • f) v = r • f v := rfl


instance instAddMonoid : AddMonoid (α →ᵇ β) :=
  DFunLike.coe_injective.addMonoid _ coe_zero coe_add fun _ _ => coe_nsmul _ _


/-- Coercion of a `NormedAddGroupHom` is an `AddMonoidHom`. Similar to `AddMonoidHom.coeFn`. -/
@[simps]
def coeFnAddHom : (α →ᵇ β) →+ α → β where
  toFun := (⇑)
  map_zero' := coe_zero
  map_add' := coe_add


/-- The additive map forgetting that a bounded continuous function is bounded. -/
@[simps]
def toContinuousMapAddHom : (α →ᵇ β) →+ C(α, β) where
  toFun := toContinuousMap
  map_zero' := rfl
  map_add' := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : PseudoMetricSpace β
      inst✝² : AddMonoid β
      inst✝¹ : BoundedAdd β
      inst✝ : ContinuousAdd β
      f g : BoundedContinuousFunction α β
      x : α
      C : Real
      ⊢ ∀ (x y : BoundedContinuousFunction α β), Eq ({ toFun := BoundedContinuousFun …
    -/
    intros
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : PseudoMetricSpace β
      inst✝² : AddMonoid β
      inst✝¹ : BoundedAdd β
      inst✝ : ContinuousAdd β
      f g : BoundedContinuousFunction α β
      x : α
      C : Real
      x✝ y✝ : BoundedContinuousFunction α β
      ⊢ Eq ({ toFun := BoundedContinuousFunction.toContinuousMap, map_zero' := ⋯ }.t …
    -/
    ext
    /-
      case h
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      inst✝⁴ : TopologicalSpace α
      inst✝³ : PseudoMetricSpace β
      inst✝² : AddMonoid β
      inst✝¹ : BoundedAdd β
      inst✝ : ContinuousAdd β
      f g : BoundedContinuousFunction α β
      x : α
      C : Real
      x✝ y✝ : BoundedContinuousFunction α β
      a✝ : α
      ⊢ Eq (({ toFun := BoundedContinuousFunction.toContinuousMap, map_zero' := ⋯ }. …
    -/
    simp
    /-
      🎉 no goals
    -/


@[to_additive]
instance instAddCommMonoid : AddCommMonoid (α →ᵇ β) where
                     /-
                       F : Type u_1
                       α : Type u
                       β : Type v
                       γ : Type w
                       inst✝⁴ : TopologicalSpace α
                       inst✝³ : PseudoMetricSpace β
                       inst✝² : AddCommMonoid β
                       inst✝¹ : BoundedAdd β
                       inst✝ : ContinuousAdd β
                       f g : BoundedContinuousFunction α β
                       ⊢ Eq (HAdd.hAdd f g) (HAdd.hAdd g f)
                     -/
  add_comm f g := by ext; simp [add_comm]
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem coe_sum {ι : Type*} (s : Finset ι) (f : ι → α →ᵇ β) :
    ⇑(∑ i ∈ s, f i) = ∑ i ∈ s, (f i : α → β) :=
  map_sum coeFnAddHom f s


theorem sum_apply {ι : Type*} (s : Finset ι) (f : ι → α →ᵇ β) (a : α) :
                                            /-
                                              α : Type u
                                              β : Type v
                                              inst✝⁴ : TopologicalSpace α
                                              inst✝³ : PseudoMetricSpace β
                                              inst✝² : AddCommMonoid β
                                              inst✝¹ : BoundedAdd β
                                              inst✝ : ContinuousAdd β
                                              ι : Type u_2
                                              s : Finset ι
                                              f : ι → BoundedContinuousFunction α β
                                              a : α
                                              ⊢ Eq ((s.sum fun i => f i) a) (s.sum fun i => (f i) a)
                                            -/
    (∑ i ∈ s, f i) a = ∑ i ∈ s, f i a := by simp
                                            /-
                                              🎉 no goals
                                            -/


instance instLipschitzAdd : LipschitzAdd (α →ᵇ β) where
  lipschitz_add :=
    ⟨LipschitzAdd.C β, by
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x : α
        C : Real
        ⊢ LipschitzWith (LipschitzAdd.C β) fun p => HAdd.hAdd p.1 p.2
      -/
      have C_nonneg := (LipschitzAdd.C β).coe_nonneg
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        ⊢ LipschitzWith (LipschitzAdd.C β) fun p => HAdd.hAdd p.1 p.2
      -/
      rw [lipschitzWith_iff_dist_le_mul]
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        ⊢ ∀ (x y : Prod (BoundedContinuousFunction α β) (BoundedContinuousFunction α β …
      -/
      rintro ⟨f₁, g₁⟩ ⟨f₂, g₂⟩
      /-
        case mk.mk
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        f₁ g₁ f₂ g₂ : BoundedContinuousFunction α β
        ⊢ LE.le (Dist.dist (HAdd.hAdd { fst := f₁, snd := g₁ }.1 { fst := f₁, snd := g …
      -/
      rw [dist_le (mul_nonneg C_nonneg dist_nonneg)]
      /-
        case mk.mk
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        f₁ g₁ f₂ g₂ : BoundedContinuousFunction α β
        ⊢ ∀ (x : α), LE.le (Dist.dist ((HAdd.hAdd { fst := f₁, snd := g₁ }.1 { fst :=  …
      -/
      intro x
      /-
        case mk.mk
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x✝ : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        f₁ g₁ f₂ g₂ : BoundedContinuousFunction α β
        x : α
        ⊢ LE.le (Dist.dist ((HAdd.hAdd { fst := f₁, snd := g₁ }.1 { fst := f₁, snd :=  …
      -/
      refine le_trans (lipschitz_with_lipschitz_const_add ⟨f₁ x, g₁ x⟩ ⟨f₂ x, g₂ x⟩) ?_
      /-
        case mk.mk
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x✝ : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        f₁ g₁ f₂ g₂ : BoundedContinuousFunction α β
        x : α
        ⊢ LE.le (HMul.hMul (↑(LipschitzAdd.C β)) (Dist.dist { fst := f₁ x, snd := g₁ x …
      -/
      refine mul_le_mul_of_nonneg_left ?_ C_nonneg
      /-
        case mk.mk
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝³ : TopologicalSpace α
        inst✝² : PseudoMetricSpace β
        inst✝¹ : AddMonoid β
        inst✝ : LipschitzAdd β
        f g : BoundedContinuousFunction α β
        x✝ : α
        C : Real
        C_nonneg : LE.le 0 ↑(LipschitzAdd.C β)
        f₁ g₁ f₂ g₂ : BoundedContinuousFunction α β
        x : α
        ⊢ LE.le (Dist.dist { fst := f₁ x, snd := g₁ x } { fst := f₂ x, snd := g₂ x })  …
      -/
                           /-
                             🎉 no goals
                           -/
      apply max_le_max <;> exact dist_coe_le_dist x⟩
                           /-
                             🎉 no goals
                           -/


/-- The pointwise difference of two bounded continuous functions is again bounded continuous. -/
instance instSub : Sub (α →ᵇ R) where
  sub f g :=
    { toFun := fun x ↦ (f x - g x),
      map_bounded' := sub_bounded_of_bounded_of_bounded f.map_bounded' g.map_bounded' }


theorem sub_apply {x : α} : (f - g) x = f x - g x := rfl


@[simp]
theorem coe_sub : ⇑(f - g) = f - g := rfl


instance [NatCast β] : NatCast (α →ᵇ β) := ⟨fun n ↦ BoundedContinuousFunction.const _ n⟩


@[simp]
theorem natCast_apply [NatCast β] (n : ℕ) (x : α) : (n : α →ᵇ β) x = n := rfl


instance [IntCast β] : IntCast (α →ᵇ β) := ⟨fun m ↦ BoundedContinuousFunction.const _ m⟩


@[simp]
theorem intCast_apply [IntCast β] (m : ℤ) (x : α) : (m : α →ᵇ β) x = m := rfl


instance instMul [Mul R] [BoundedMul R] [ContinuousMul R] :
    Mul (α →ᵇ R) where
  mul f g :=
    { toFun := fun x ↦ f x * g x
      continuous_toFun := f.continuous.mul g.continuous
      map_bounded' := mul_bounded_of_bounded_of_bounded (map_bounded f) (map_bounded g) }


@[simp]
theorem coe_mul [Mul R] [BoundedMul R] [ContinuousMul R] (f g : α →ᵇ R) : ⇑(f * g) = f * g := rfl


theorem mul_apply [Mul R] [BoundedMul R] [ContinuousMul R] (f g : α →ᵇ R) (x : α) :
    (f * g) x = f x * g x := rfl


instance instPow [Monoid R] [BoundedMul R] [ContinuousMul R] : Pow (α →ᵇ R) ℕ where
  pow f n :=
    { toFun := fun x ↦ (f x) ^ n
      continuous_toFun := f.continuous.pow n
      map_bounded' := by
        /-
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝⁴ : TopologicalSpace α
          R : Type u_2
          inst✝³ : PseudoMetricSpace R
          inst✝² : Monoid R
          inst✝¹ : BoundedMul R
          inst✝ : ContinuousMul R
          f : BoundedContinuousFunction α R
          n : Nat
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := fun x => HPow.hPow …
        -/
        obtain ⟨C, hC⟩ := Metric.isBounded_iff.mp <| isBounded_pow (isBounded_range f) n
        /-
          case intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝⁴ : TopologicalSpace α
          R : Type u_2
          inst✝³ : PseudoMetricSpace R
          inst✝² : Monoid R
          inst✝¹ : BoundedMul R
          inst✝ : ContinuousMul R
          f : BoundedContinuousFunction α R
          n : Nat
          C : Real
          hC : ∀ ⦃x : R⦄, Membership.mem (Set.image (fun x => HPow.hPow x n) (Set.range  …
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := fun x => HPow.hPow …
        -/
        exact ⟨C, fun x y ↦ hC (by simp) (by simp)⟩ }
        /-
          🎉 no goals
        -/


theorem coe_pow [Monoid R] [BoundedMul R] [ContinuousMul R] (n : ℕ) (f : α →ᵇ R) :
    ⇑(f ^ n) = (⇑f) ^ n := rfl


@[simp]
theorem pow_apply [Monoid R] [BoundedMul R] [ContinuousMul R] (n : ℕ) (f : α →ᵇ R) (x : α) :
    (f ^ n) x = f x ^ n := rfl


instance instMonoid [Monoid R] [BoundedMul R] [ContinuousMul R] :
    Monoid (α →ᵇ R) :=
  Injective.monoid _ DFunLike.coe_injective' rfl (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)


instance instCommMonoid [CommMonoid R] [BoundedMul R] [ContinuousMul R] :
    CommMonoid (α →ᵇ R) where
  __ := instMonoid
                     /-
                       F : Type u_1
                       α : Type u
                       β : Type v
                       γ : Type w
                       inst✝⁴ : TopologicalSpace α
                       R : Type u_2
                       inst✝³ : PseudoMetricSpace R
                       inst✝² : CommMonoid R
                       inst✝¹ : BoundedMul R
                       inst✝ : ContinuousMul R
                       f g : BoundedContinuousFunction α R
                       ⊢ Eq (HMul.hMul f g) (HMul.hMul g f)
                     -/
  mul_comm f g := by ext x; simp [mul_apply, mul_comm]
                            /-
                              🎉 no goals
                            -/


instance instSemiring [Semiring R] [BoundedMul R] [ContinuousMul R]
    [BoundedAdd R] [ContinuousAdd R] :
    Semiring (α →ᵇ R) :=
  Injective.semiring _ DFunLike.coe_injective'
    rfl rfl (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ ↦ rfl)


instance instNorm : Norm (α →ᵇ β) := ⟨(dist · 0)⟩


theorem norm_def : ‖f‖ = dist f 0 := rfl


/-- The norm of a bounded continuous function is the supremum of `‖f x‖`.
We use `sInf` to ensure that the definition works if `α` has no elements. -/
theorem norm_eq (f : α →ᵇ β) : ‖f‖ = sInf { C : ℝ | 0 ≤ C ∧ ∀ x : α, ‖f x‖ ≤ C } := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    ⊢ Eq (Norm.norm f) (InfSet.sInf (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE …
  -/
  simp [norm_def, BoundedContinuousFunction.dist_eq]
  /-
    🎉 no goals
  -/


/-- When the domain is non-empty, we do not need the `0 ≤ C` condition in the formula for `‖f‖` as a
`sInf`. -/
theorem norm_eq_of_nonempty [h : Nonempty α] : ‖f‖ = sInf { C : ℝ | ∀ x : α, ‖f x‖ ≤ C } := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    h : Nonempty α
    ⊢ Eq (Norm.norm f) (InfSet.sInf (setOf fun C => ∀ (x : α), LE.le (Norm.norm (f …
  -/
  obtain ⟨a⟩ := h
  /-
    case intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    a : α
    ⊢ Eq (Norm.norm f) (InfSet.sInf (setOf fun C => ∀ (x : α), LE.le (Norm.norm (f …
  -/
  rw [norm_eq]
  /-
    case intro
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    a : α
    ⊢ Eq (InfSet.sInf (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Norm.norm …
  -/
  congr
  /-
    case intro.e_a
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    a : α
    ⊢ Eq (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Norm.norm (f x)) C)) ( …
  -/
  ext
  /-
    case intro.e_a.h
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    a : α
    x✝ : Real
    ⊢ Iff (Membership.mem (setOf fun C => And (LE.le 0 C) (∀ (x : α), LE.le (Norm. …
  -/
  simp only [mem_setOf_eq, and_iff_right_iff_imp]
  /-
    case intro.e_a.h
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    a : α
    x✝ : Real
    ⊢ (∀ (x : α), LE.le (Norm.norm (f x)) x✝) → LE.le 0 x✝
  -/
  exact fun h' => le_trans (norm_nonneg (f a)) (h' a)
  /-
    🎉 no goals
  -/


@[simp]
theorem norm_eq_zero_of_empty [IsEmpty α] : ‖f‖ = 0 :=
  dist_zero_of_empty


theorem norm_coe_le_norm (x : α) : ‖f x‖ ≤ ‖f‖ :=
  calc
                                              /-
                                                α : Type u
                                                β : Type v
                                                inst✝¹ : TopologicalSpace α
                                                inst✝ : SeminormedAddCommGroup β
                                                f : BoundedContinuousFunction α β
                                                x : α
                                                ⊢ Eq (Norm.norm (f x)) (Dist.dist (f x) (0 x))
                                              -/
    ‖f x‖ = dist (f x) ((0 : α →ᵇ β) x) := by simp [dist_zero_right]
                                              /-
                                                🎉 no goals
                                              -/
    _ ≤ ‖f‖ := dist_coe_le_dist _


lemma neg_norm_le_apply (f : α →ᵇ ℝ) (x : α) :
    -‖f‖ ≤ f x := (abs_le.mp (norm_coe_le_norm f x)).1


lemma apply_le_norm (f : α →ᵇ ℝ) (x : α) :
    f x ≤ ‖f‖ := (abs_le.mp (norm_coe_le_norm f x)).2


theorem dist_le_two_norm' {f : γ → β} {C : ℝ} (hC : ∀ x, ‖f x‖ ≤ C) (x y : γ) :
    dist (f x) (f y) ≤ 2 * C :=
  calc
    dist (f x) (f y) ≤ ‖f x‖ + ‖f y‖ := dist_le_norm_add_norm _ _
    _ ≤ C + C := add_le_add (hC x) (hC y)
    _ = 2 * C := (two_mul _).symm


/-- Distance between the images of any two points is at most twice the norm of the function. -/
theorem dist_le_two_norm (x y : α) : dist (f x) (f y) ≤ 2 * ‖f‖ :=
  dist_le_two_norm' f.norm_coe_le_norm x y


/-- The norm of a function is controlled by the supremum of the pointwise norms. -/
theorem norm_le (C0 : (0 : ℝ) ≤ C) : ‖f‖ ≤ C ↔ ∀ x : α, ‖f x‖ ≤ C := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    C : Real
    C0 : LE.le 0 C
    ⊢ Iff (LE.le (Norm.norm f) C) (∀ (x : α), LE.le (Norm.norm (f x)) C)
  -/
  simpa using @dist_le _ _ _ _ f 0 _ C0
  /-
    🎉 no goals
  -/


theorem norm_le_of_nonempty [Nonempty α] {f : α →ᵇ β} {M : ℝ} : ‖f‖ ≤ M ↔ ∀ x, ‖f x‖ ≤ M := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : Nonempty α
    f : BoundedContinuousFunction α β
    M : Real
    ⊢ Iff (LE.le (Norm.norm f) M) (∀ (x : α), LE.le (Norm.norm (f x)) M)
  -/
  simp_rw [norm_def, ← dist_zero_right]
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : Nonempty α
    f : BoundedContinuousFunction α β
    M : Real
    ⊢ Iff (LE.le (Dist.dist f 0) M) (∀ (x : α), LE.le (Dist.dist (f x) 0) M)
  -/
  exact dist_le_iff_of_nonempty
  /-
    🎉 no goals
  -/


theorem norm_lt_iff_of_compact [CompactSpace α] {f : α →ᵇ β} {M : ℝ} (M0 : 0 < M) :
    ‖f‖ < M ↔ ∀ x, ‖f x‖ < M := by
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : CompactSpace α
    f : BoundedContinuousFunction α β
    M : Real
    M0 : LT.lt 0 M
    ⊢ Iff (LT.lt (Norm.norm f) M) (∀ (x : α), LT.lt (Norm.norm (f x)) M)
  -/
  simp_rw [norm_def, ← dist_zero_right]
  /-
    α : Type u
    β : Type v
    inst✝² : TopologicalSpace α
    inst✝¹ : SeminormedAddCommGroup β
    inst✝ : CompactSpace α
    f : BoundedContinuousFunction α β
    M : Real
    M0 : LT.lt 0 M
    ⊢ Iff (LT.lt (Dist.dist f 0) M) (∀ (x : α), LT.lt (Dist.dist (f x) 0) M)
  -/
  exact dist_lt_iff_of_compact M0
  /-
    🎉 no goals
  -/


theorem norm_lt_iff_of_nonempty_compact [Nonempty α] [CompactSpace α] {f : α →ᵇ β} {M : ℝ} :
    ‖f‖ < M ↔ ∀ x, ‖f x‖ < M := by
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : SeminormedAddCommGroup β
    inst✝¹ : Nonempty α
    inst✝ : CompactSpace α
    f : BoundedContinuousFunction α β
    M : Real
    ⊢ Iff (LT.lt (Norm.norm f) M) (∀ (x : α), LT.lt (Norm.norm (f x)) M)
  -/
  simp_rw [norm_def, ← dist_zero_right]
  /-
    α : Type u
    β : Type v
    inst✝³ : TopologicalSpace α
    inst✝² : SeminormedAddCommGroup β
    inst✝¹ : Nonempty α
    inst✝ : CompactSpace α
    f : BoundedContinuousFunction α β
    M : Real
    ⊢ Iff (LT.lt (Dist.dist f 0) M) (∀ (x : α), LT.lt (Dist.dist (f x) 0) M)
  -/
  exact dist_lt_iff_of_nonempty_compact
  /-
    🎉 no goals
  -/


/-- Norm of `const α b` is less than or equal to `‖b‖`. If `α` is nonempty,
then it is equal to `‖b‖`. -/
theorem norm_const_le (b : β) : ‖const α b‖ ≤ ‖b‖ :=
  (norm_le (norm_nonneg b)).2 fun _ => le_rfl


@[simp]
theorem norm_const_eq [h : Nonempty α] (b : β) : ‖const α b‖ = ‖b‖ :=
  le_antisymm (norm_const_le b) <| h.elim fun x => (const α b).norm_coe_le_norm x


/-- Constructing a bounded continuous function from a uniformly bounded continuous
function taking values in a normed group. -/
def ofNormedAddCommGroup {α : Type u} {β : Type v} [TopologicalSpace α] [SeminormedAddCommGroup β]
    (f : α → β) (Hf : Continuous f) (C : ℝ) (H : ∀ x, ‖f x‖ ≤ C) : α →ᵇ β :=
  ⟨⟨fun n => f n, Hf⟩, ⟨_, dist_le_two_norm' H⟩⟩


@[simp]
theorem coe_ofNormedAddCommGroup {α : Type u} {β : Type v} [TopologicalSpace α]
    [SeminormedAddCommGroup β] (f : α → β) (Hf : Continuous f) (C : ℝ) (H : ∀ x, ‖f x‖ ≤ C) :
    (ofNormedAddCommGroup f Hf C H : α → β) = f := rfl


theorem norm_ofNormedAddCommGroup_le {f : α → β} (hfc : Continuous f) {C : ℝ} (hC : 0 ≤ C)
    (hfC : ∀ x, ‖f x‖ ≤ C) : ‖ofNormedAddCommGroup f hfc C hfC‖ ≤ C :=
  (norm_le hC).2 hfC


/-- Constructing a bounded continuous function from a uniformly bounded
function on a discrete space, taking values in a normed group. -/
def ofNormedAddCommGroupDiscrete {α : Type u} {β : Type v} [TopologicalSpace α] [DiscreteTopology α]
    [SeminormedAddCommGroup β] (f : α → β) (C : ℝ) (H : ∀ x, norm (f x) ≤ C) : α →ᵇ β :=
  ofNormedAddCommGroup f continuous_of_discreteTopology C H


@[simp]
theorem coe_ofNormedAddCommGroupDiscrete {α : Type u} {β : Type v} [TopologicalSpace α]
    [DiscreteTopology α] [SeminormedAddCommGroup β] (f : α → β) (C : ℝ) (H : ∀ x, ‖f x‖ ≤ C) :
    (ofNormedAddCommGroupDiscrete f C H : α → β) = f := rfl


/-- Taking the pointwise norm of a bounded continuous function with values in a
`SeminormedAddCommGroup` yields a bounded continuous function with values in ℝ. -/
def normComp : α →ᵇ ℝ :=
  f.comp norm lipschitzWith_one_norm


@[simp]
theorem coe_normComp : (f.normComp : α → ℝ) = norm ∘ f := rfl


@[simp]
theorem norm_normComp : ‖f.normComp‖ = ‖f‖ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    ⊢ Eq (Norm.norm f.normComp) (Norm.norm f)
  -/
  simp only [norm_eq, coe_normComp, norm_norm, Function.comp]
  /-
    🎉 no goals
  -/


theorem bddAbove_range_norm_comp : BddAbove <| Set.range <| norm ∘ f :=
  (@isBounded_range _ _ _ _ f.normComp).bddAbove


theorem norm_eq_iSup_norm : ‖f‖ = ⨆ x : α, ‖f x‖ := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f : BoundedContinuousFunction α β
    ⊢ Eq (Norm.norm f) (iSup fun x => Norm.norm (f x))
  -/
  simp_rw [norm_def, dist_eq_iSup, coe_zero, Pi.zero_apply, dist_zero_right]
  /-
    🎉 no goals
  -/


/-- If `‖(1 : β)‖ = 1`, then `‖(1 : α →ᵇ β)‖ = 1` if `α` is nonempty. -/
instance instNormOneClass [Nonempty α] [One β] [NormOneClass β] : NormOneClass (α →ᵇ β) where
                 /-
                   F : Type u_1
                   α : Type u
                   β : Type v
                   γ : Type w
                   inst✝⁴ : TopologicalSpace α
                   inst✝³ : SeminormedAddCommGroup β
                   f g : BoundedContinuousFunction α β
                   x : α
                   C : Real
                   inst✝² : Nonempty α
                   inst✝¹ : One β
                   inst✝ : NormOneClass β
                   ⊢ Eq (Norm.norm 1) 1
                 -/
  norm_one := by simp only [norm_eq_iSup_norm, coe_one, Pi.one_apply, norm_one, ciSup_const]
                 /-
                   🎉 no goals
                 -/


/-- The pointwise opposite of a bounded continuous function is again bounded continuous. -/
instance : Neg (α →ᵇ β) :=
  ⟨fun f =>
    ofNormedAddCommGroup (-f) f.continuous.neg ‖f‖ fun x =>
      norm_neg ((⇑f) x) ▸ f.norm_coe_le_norm x⟩


@[simp]
theorem coe_neg : ⇑(-f) = -f := rfl


theorem neg_apply : (-f) x = -f x := rfl


@[simp]
theorem mkOfCompact_neg [CompactSpace α] (f : C(α, β)) : mkOfCompact (-f) = -mkOfCompact f := rfl


@[simp]
theorem mkOfCompact_sub [CompactSpace α] (f g : C(α, β)) :
    mkOfCompact (f - g) = mkOfCompact f - mkOfCompact g := rfl


@[simp]
theorem coe_zsmulRec : ∀ z, ⇑(zsmulRec (· • ·) z f) = z • ⇑f
                      /-
                        α : Type u
                        β : Type v
                        inst✝¹ : TopologicalSpace α
                        inst✝ : SeminormedAddCommGroup β
                        f : BoundedContinuousFunction α β
                        n : Nat
                        ⊢ Eq (⇑(zsmulRec (fun x1 x2 => HSMul.hSMul x1 x2) (Int.ofNat n) f)) (HSMul.hSM …
                      -/
  | Int.ofNat n => by rw [zsmulRec, Int.ofNat_eq_coe, coe_nsmul, natCast_zsmul]
                      /-
                        🎉 no goals
                      -/
                        /-
                          α : Type u
                          β : Type v
                          inst✝¹ : TopologicalSpace α
                          inst✝ : SeminormedAddCommGroup β
                          f : BoundedContinuousFunction α β
                          n : Nat
                          ⊢ Eq (⇑(zsmulRec (fun x1 x2 => HSMul.hSMul x1 x2) (Int.negSucc n) f)) (HSMul.h …
                        -/
  | Int.negSucc n => by rw [zsmulRec, negSucc_zsmul, coe_neg, coe_nsmul]
                        /-
                          🎉 no goals
                        -/


instance instSMulInt : SMul ℤ (α →ᵇ β) where
  smul n f :=
    { toContinuousMap := n • f.toContinuousMap
                         /-
                           F : Type u_1
                           α : Type u
                           β : Type v
                           γ : Type w
                           inst✝¹ : TopologicalSpace α
                           inst✝ : SeminormedAddCommGroup β
                           f✝ g : BoundedContinuousFunction α β
                           x : α
                           C : Real
                           n : Int
                           f : BoundedContinuousFunction α β
                           ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ((HSMul.hSMul n f.toContinuous …
                         -/
      map_bounded' := by simpa using (zsmulRec (· • ·) n f).map_bounded' }
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem coe_zsmul (r : ℤ) (f : α →ᵇ β) : ⇑(r • f) = r • ⇑f := rfl


@[simp]
theorem zsmul_apply (r : ℤ) (f : α →ᵇ β) (v : α) : (r • f) v = r • f v := rfl


instance instAddCommGroup : AddCommGroup (α →ᵇ β) :=
  DFunLike.coe_injective.addCommGroup _ coe_zero coe_add coe_neg coe_sub (fun _ _ => coe_nsmul _ _)
    fun _ _ => coe_zsmul _ _


instance instSeminormedAddCommGroup : SeminormedAddCommGroup (α →ᵇ β) where
                    /-
                      F : Type u_1
                      α : Type u
                      β : Type v
                      γ : Type w
                      inst✝¹ : TopologicalSpace α
                      inst✝ : SeminormedAddCommGroup β
                      f✝ g✝ : BoundedContinuousFunction α β
                      x : α
                      C : Real
                      f g : BoundedContinuousFunction α β
                      ⊢ Eq (Dist.dist f g) (Norm.norm (HSub.hSub f g))
                    -/
  dist_eq f g := by simp only [norm_eq, dist_eq, dist_eq_norm, sub_apply]
                    /-
                      🎉 no goals
                    -/


instance instNormedAddCommGroup {α β} [TopologicalSpace α] [NormedAddCommGroup β] :
    NormedAddCommGroup (α →ᵇ β) :=
  { instSeminormedAddCommGroup with
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): Added a proof for `eq_of_dist_eq_zero`
    eq_of_dist_eq_zero }


theorem nnnorm_def : ‖f‖₊ = nndist f 0 := rfl


theorem nnnorm_coe_le_nnnorm (x : α) : ‖f x‖₊ ≤ ‖f‖₊ :=
  norm_coe_le_norm _ _


theorem nndist_le_two_nnnorm (x y : α) : nndist (f x) (f y) ≤ 2 * ‖f‖₊ :=
  dist_le_two_norm _ _ _


/-- The `nnnorm` of a function is controlled by the supremum of the pointwise `nnnorm`s. -/
theorem nnnorm_le (C : ℝ≥0) : ‖f‖₊ ≤ C ↔ ∀ x : α, ‖f x‖₊ ≤ C :=
  norm_le C.prop


theorem nnnorm_const_le (b : β) : ‖const α b‖₊ ≤ ‖b‖₊ :=
  norm_const_le _


@[simp]
theorem nnnorm_const_eq [Nonempty α] (b : β) : ‖const α b‖₊ = ‖b‖₊ :=
  Subtype.ext <| norm_const_eq _


theorem nnnorm_eq_iSup_nnnorm : ‖f‖₊ = ⨆ x : α, ‖f x‖₊ :=
                                                   /-
                                                     α : Type u
                                                     β : Type v
                                                     inst✝¹ : TopologicalSpace α
                                                     inst✝ : SeminormedAddCommGroup β
                                                     f : BoundedContinuousFunction α β
                                                     ⊢ Eq (iSup fun x => Norm.norm (f x)) ↑(iSup fun x => NNNorm.nnnorm (f x))
                                                   -/
  Subtype.ext <| (norm_eq_iSup_norm f).trans <| by simp_rw [val_eq_coe, NNReal.coe_iSup, coe_nnnorm]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem abs_diff_coe_le_dist : ‖f x - g x‖ ≤ dist f g := by
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f g : BoundedContinuousFunction α β
    x : α
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (g x))) (Dist.dist f g)
  -/
  rw [dist_eq_norm]
  /-
    α : Type u
    β : Type v
    inst✝¹ : TopologicalSpace α
    inst✝ : SeminormedAddCommGroup β
    f g : BoundedContinuousFunction α β
    x : α
    ⊢ LE.le (Norm.norm (HSub.hSub (f x) (g x))) (Norm.norm (HSub.hSub f g))
  -/
  exact (f - g).norm_coe_le_norm x
  /-
    🎉 no goals
  -/


theorem coe_le_coe_add_dist {f g : α →ᵇ ℝ} : f x ≤ g x + dist f g :=
  sub_le_iff_le_add'.1 <| (abs_le.1 <| @dist_coe_le_dist _ _ _ _ f g x).2


theorem norm_compContinuous_le [TopologicalSpace γ] (f : α →ᵇ β) (g : C(γ, α)) :
    ‖f.compContinuous g‖ ≤ ‖f‖ :=
  ((lipschitz_compContinuous g).dist_le_mul f 0).trans <| by
    /-
      α : Type u
      β : Type v
      γ : Type w
      inst✝² : TopologicalSpace α
      inst✝¹ : SeminormedAddCommGroup β
      inst✝ : TopologicalSpace γ
      f : BoundedContinuousFunction α β
      g : ContinuousMap γ α
      ⊢ LE.le (HMul.hMul (↑1) (Dist.dist f 0)) (Norm.norm f)
    -/
    rw [NNReal.coe_one, one_mul, dist_zero_right]
    /-
      🎉 no goals
    -/


instance instSMul : SMul 𝕜 (α →ᵇ β) where
  smul c f :=
    { toContinuousMap := c • f.toContinuousMap
      map_bounded' :=
        let ⟨b, hb⟩ := f.bounded
        ⟨dist c 0 * b, fun x y => by
          /-
            F : Type u_1
            α : Type u
            β : Type v
            γ : Type w
            𝕜 : Type u_2
            inst✝⁶ : PseudoMetricSpace 𝕜
            inst✝⁵ : TopologicalSpace α
            inst✝⁴ : PseudoMetricSpace β
            inst✝³ : Zero 𝕜
            inst✝² : Zero β
            inst✝¹ : SMul 𝕜 β
            inst✝ : BoundedSMul 𝕜 β
            c : 𝕜
            f : BoundedContinuousFunction α β
            b : Real
            hb : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) b
            x y : α
            ⊢ LE.le (Dist.dist ((HSMul.hSMul c f.toContinuousMap).toFun x) ((HSMul.hSMul c …
          -/
          refine (dist_smul_pair c (f x) (f y)).trans ?_
          /-
            F : Type u_1
            α : Type u
            β : Type v
            γ : Type w
            𝕜 : Type u_2
            inst✝⁶ : PseudoMetricSpace 𝕜
            inst✝⁵ : TopologicalSpace α
            inst✝⁴ : PseudoMetricSpace β
            inst✝³ : Zero 𝕜
            inst✝² : Zero β
            inst✝¹ : SMul 𝕜 β
            inst✝ : BoundedSMul 𝕜 β
            c : 𝕜
            f : BoundedContinuousFunction α β
            b : Real
            hb : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) b
            x y : α
            ⊢ LE.le (HMul.hMul (Dist.dist c 0) (Dist.dist (f x) (f y))) (HMul.hMul (Dist.d …
          -/
          refine mul_le_mul_of_nonneg_left ?_ dist_nonneg
          /-
            F : Type u_1
            α : Type u
            β : Type v
            γ : Type w
            𝕜 : Type u_2
            inst✝⁶ : PseudoMetricSpace 𝕜
            inst✝⁵ : TopologicalSpace α
            inst✝⁴ : PseudoMetricSpace β
            inst✝³ : Zero 𝕜
            inst✝² : Zero β
            inst✝¹ : SMul 𝕜 β
            inst✝ : BoundedSMul 𝕜 β
            c : 𝕜
            f : BoundedContinuousFunction α β
            b : Real
            hb : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) b
            x y : α
            ⊢ LE.le (Dist.dist (f x) (f y)) b
          -/
          exact hb x y⟩ }
          /-
            🎉 no goals
          -/


@[simp]
theorem coe_smul (c : 𝕜) (f : α →ᵇ β) : ⇑(c • f) = fun x => c • f x := rfl


theorem smul_apply (c : 𝕜) (f : α →ᵇ β) (x : α) : (c • f) x = c • f x := rfl


instance instIsScalarTower {𝕜' : Type*} [PseudoMetricSpace 𝕜'] [Zero 𝕜'] [SMul 𝕜' β]
    [BoundedSMul 𝕜' β] [SMul 𝕜' 𝕜] [IsScalarTower 𝕜' 𝕜 β] :
    IsScalarTower 𝕜' 𝕜 (α →ᵇ β) where
  smul_assoc _ _ _ := ext fun _ ↦ smul_assoc ..


instance instSMulCommClass {𝕜' : Type*} [PseudoMetricSpace 𝕜'] [Zero 𝕜'] [SMul 𝕜' β]
    [BoundedSMul 𝕜' β] [SMulCommClass 𝕜' 𝕜 β] :
    SMulCommClass 𝕜' 𝕜 (α →ᵇ β) where
  smul_comm _ _ _ := ext fun _ ↦ smul_comm ..


instance instIsCentralScalar [SMul 𝕜ᵐᵒᵖ β] [IsCentralScalar 𝕜 β] : IsCentralScalar 𝕜 (α →ᵇ β) where
  op_smul_eq_smul _ _ := ext fun _ => op_smul_eq_smul _ _


instance instBoundedSMul : BoundedSMul 𝕜 (α →ᵇ β) where
  dist_smul_pair' c f₁ f₂ := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c : 𝕜
      f₁ f₂ : BoundedContinuousFunction α β
      ⊢ LE.le (Dist.dist (HSMul.hSMul c f₁) (HSMul.hSMul c f₂)) (HMul.hMul (Dist.dis …
    -/
    rw [dist_le (mul_nonneg dist_nonneg dist_nonneg)]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c : 𝕜
      f₁ f₂ : BoundedContinuousFunction α β
      ⊢ ∀ (x : α), LE.le (Dist.dist ((HSMul.hSMul c f₁) x) ((HSMul.hSMul c f₂) x)) ( …
    -/
    intro x
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c : 𝕜
      f₁ f₂ : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (Dist.dist ((HSMul.hSMul c f₁) x) ((HSMul.hSMul c f₂) x)) (HMul.hMul ( …
    -/
    refine (dist_smul_pair c (f₁ x) (f₂ x)).trans ?_
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c : 𝕜
      f₁ f₂ : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (HMul.hMul (Dist.dist c 0) (Dist.dist (f₁ x) (f₂ x))) (HMul.hMul (Dist …
    -/
    exact mul_le_mul_of_nonneg_left (dist_coe_le_dist x) dist_nonneg
    /-
      🎉 no goals
    -/
  dist_pair_smul' c₁ c₂ f := by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c₁ c₂ : 𝕜
      f : BoundedContinuousFunction α β
      ⊢ LE.le (Dist.dist (HSMul.hSMul c₁ f) (HSMul.hSMul c₂ f)) (HMul.hMul (Dist.dis …
    -/
    rw [dist_le (mul_nonneg dist_nonneg dist_nonneg)]
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c₁ c₂ : 𝕜
      f : BoundedContinuousFunction α β
      ⊢ ∀ (x : α), LE.le (Dist.dist ((HSMul.hSMul c₁ f) x) ((HSMul.hSMul c₂ f) x)) ( …
    -/
    intro x
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c₁ c₂ : 𝕜
      f : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (Dist.dist ((HSMul.hSMul c₁ f) x) ((HSMul.hSMul c₂ f) x)) (HMul.hMul ( …
    -/
    refine (dist_pair_smul c₁ c₂ (f x)).trans ?_
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c₁ c₂ : 𝕜
      f : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (HMul.hMul (Dist.dist c₁ c₂) (Dist.dist (f x) 0)) (HMul.hMul (Dist.dis …
    -/
    refine mul_le_mul_of_nonneg_left ?_ dist_nonneg
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c₁ c₂ : 𝕜
      f : BoundedContinuousFunction α β
      x : α
      ⊢ LE.le (Dist.dist (f x) 0) (Dist.dist f 0)
    -/
    convert dist_coe_le_dist (β := β) x
    /-
      case h.e'_3.h.e'_4
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝⁶ : PseudoMetricSpace 𝕜
      inst✝⁵ : TopologicalSpace α
      inst✝⁴ : PseudoMetricSpace β
      inst✝³ : Zero 𝕜
      inst✝² : Zero β
      inst✝¹ : SMul 𝕜 β
      inst✝ : BoundedSMul 𝕜 β
      c₁ c₂ : 𝕜
      f : BoundedContinuousFunction α β
      x : α
      ⊢ Eq 0 (0 x)
    -/
    simp
    /-
      🎉 no goals
    -/


instance instMulAction : MulAction 𝕜 (α →ᵇ β) :=
  DFunLike.coe_injective.mulAction _ coe_smul


instance instDistribMulAction : DistribMulAction 𝕜 (α →ᵇ β) :=
  DFunLike.coe_injective.distribMulAction ⟨⟨_, coe_zero⟩, coe_add⟩ coe_smul


instance instModule : Module 𝕜 (α →ᵇ β) :=
  DFunLike.coe_injective.module _ ⟨⟨_, coe_zero⟩, coe_add⟩  coe_smul


/-- The evaluation at a point, as a continuous linear map from `α →ᵇ β` to `β`. -/
@[simps]
def evalCLM (x : α) : (α →ᵇ β) →L[𝕜] β where
  toFun f := f x
  map_add' _ _ := add_apply _ _
  map_smul' _ _ := smul_apply _ _ _


/-- The linear map forgetting that a bounded continuous function is bounded. -/
@[simps]
def toContinuousMapLinearMap : (α →ᵇ β) →ₗ[𝕜] C(α, β) where
  toFun := toContinuousMap
  map_smul' _ _ := rfl
  map_add' _ _ := rfl


instance instNormedSpace [NormedField 𝕜] [NormedSpace 𝕜 β] : NormedSpace 𝕜 (α →ᵇ β) :=
  ⟨fun c f => by
    /-
      F : Type u_1
      α : Type u
      β : Type v
      γ : Type w
      𝕜 : Type u_2
      inst✝³ : TopologicalSpace α
      inst✝² : SeminormedAddCommGroup β
      f✝ g : BoundedContinuousFunction α β
      x : α
      C : Real
      inst✝¹ : NormedField 𝕜
      inst✝ : NormedSpace 𝕜 β
      c : 𝕜
      f : BoundedContinuousFunction α β
      ⊢ LE.le (Norm.norm (HSMul.hSMul c f)) (HMul.hMul (Norm.norm c) (Norm.norm f))
    -/
    refine norm_ofNormedAddCommGroup_le _ (mul_nonneg (norm_nonneg _) (norm_nonneg _)) ?_
    exact fun x =>
      norm_smul c (f x) ▸ mul_le_mul_of_nonneg_left (f.norm_coe_le_norm _) (norm_nonneg _)⟩


/-- Postcomposition of bounded continuous functions into a normed module by a continuous linear map
is a continuous linear map.
Upgraded version of `ContinuousLinearMap.compLeftContinuous`, similar to `LinearMap.compLeft`. -/
protected def _root_.ContinuousLinearMap.compLeftContinuousBounded (g : β →L[𝕜] γ) :
    (α →ᵇ β) →L[𝕜] α →ᵇ γ :=
  LinearMap.mkContinuous
    { toFun := fun f =>
        ofNormedAddCommGroup (g ∘ f) (g.continuous.comp f.continuous) (‖g‖ * ‖f‖) fun x =>
          g.le_opNorm_of_le (f.norm_coe_le_norm x)
                                /-
                                  F : Type u_1
                                  α : Type u
                                  β : Type v
                                  γ : Type w
                                  𝕜 : Type u_2
                                  inst✝⁵ : TopologicalSpace α
                                  inst✝⁴ : SeminormedAddCommGroup β
                                  f✝ g✝¹ : BoundedContinuousFunction α β
                                  x : α
                                  C : Real
                                  inst✝³ : NontriviallyNormedField 𝕜
                                  inst✝² : NormedSpace 𝕜 β
                                  inst✝¹ : SeminormedAddCommGroup γ
                                  inst✝ : NormedSpace 𝕜 γ
                                  g✝ : ContinuousLinearMap (RingHom.id 𝕜) β γ
                                  f g : BoundedContinuousFunction α β
                                  ⊢ Eq ((fun f => BoundedContinuousFunction.ofNormedAddCommGroup (Function.comp  …
                                -/
      map_add' := fun f g => by ext; simp
                                     /-
                                       🎉 no goals
                                     -/
                                 /-
                                   F : Type u_1
                                   α : Type u
                                   β : Type v
                                   γ : Type w
                                   𝕜 : Type u_2
                                   inst✝⁵ : TopologicalSpace α
                                   inst✝⁴ : SeminormedAddCommGroup β
                                   f✝ g✝ : BoundedContinuousFunction α β
                                   x : α
                                   C : Real
                                   inst✝³ : NontriviallyNormedField 𝕜
                                   inst✝² : NormedSpace 𝕜 β
                                   inst✝¹ : SeminormedAddCommGroup γ
                                   inst✝ : NormedSpace 𝕜 γ
                                   g : ContinuousLinearMap (RingHom.id 𝕜) β γ
                                   c : 𝕜
                                   f : BoundedContinuousFunction α β
                                   ⊢ Eq ({ toFun := fun f => BoundedContinuousFunction.ofNormedAddCommGroup (Func …
                                 -/
      map_smul' := fun c f => by ext; simp } ‖g‖ fun f =>
                                      /-
                                        🎉 no goals
                                      -/
        norm_ofNormedAddCommGroup_le _ (mul_nonneg (norm_nonneg g) (norm_nonneg f))
                       /-
                         F : Type u_1
                         α : Type u
                         β : Type v
                         γ : Type w
                         𝕜 : Type u_2
                         inst✝⁵ : TopologicalSpace α
                         inst✝⁴ : SeminormedAddCommGroup β
                         f✝ g✝ : BoundedContinuousFunction α β
                         x✝ : α
                         C : Real
                         inst✝³ : NontriviallyNormedField 𝕜
                         inst✝² : NormedSpace 𝕜 β
                         inst✝¹ : SeminormedAddCommGroup γ
                         inst✝ : NormedSpace 𝕜 γ
                         g : ContinuousLinearMap (RingHom.id 𝕜) β γ
                         f : BoundedContinuousFunction α β
                         x : α
                         ⊢ LE.le (Norm.norm (Function.comp (⇑g) (⇑f) x)) (HMul.hMul (Norm.norm g) (Norm …
                       -/
          (fun x => by exact g.le_opNorm_of_le (f.norm_coe_le_norm x))
                       /-
                         🎉 no goals
                       -/


@[simp]
theorem _root_.ContinuousLinearMap.compLeftContinuousBounded_apply (g : β →L[𝕜] γ) (f : α →ᵇ β)
    (x : α) : (g.compLeftContinuousBounded α f) x = g (f x) := rfl


instance instNonUnitalRing : NonUnitalRing (α →ᵇ R) :=
  DFunLike.coe_injective.nonUnitalRing _ coe_zero coe_add coe_mul coe_neg coe_sub
    (fun _ _ => coe_nsmul _ _) fun _ _ => coe_zsmul _ _


instance instNonUnitalSeminormedRing : NonUnitalSeminormedRing (α →ᵇ R) :=
  { instSeminormedAddCommGroup with
    norm_mul := fun f g =>
      norm_ofNormedAddCommGroup_le _ (mul_nonneg (norm_nonneg _) (norm_nonneg _))
        (fun x ↦ (norm_mul_le _ _).trans <|
          mul_le_mul (norm_coe_le_norm f x) (norm_coe_le_norm g x) (norm_nonneg _) (norm_nonneg _))
    -- Porting note: These 5 fields were missing. Add them.
    left_distrib, right_distrib, zero_mul, mul_zero, mul_assoc }


instance instNonUnitalSeminormedCommRing [NonUnitalSeminormedCommRing R] :
    NonUnitalSeminormedCommRing (α →ᵇ R) where
  mul_comm _ _ := ext fun _ ↦ mul_comm ..


instance instNonUnitalNormedRing [NonUnitalNormedRing R] : NonUnitalNormedRing (α →ᵇ R) where
  __ := instNonUnitalSeminormedRing
  __ := instNormedAddCommGroup


instance instNonUnitalNormedCommRing [NonUnitalNormedCommRing R] :
    NonUnitalNormedCommRing (α →ᵇ R) where
  mul_comm := mul_comm


@[simp]
theorem coe_npowRec (f : α →ᵇ R) : ∀ n, ⇑(npowRec n f) = (⇑f) ^ n
            /-
              α : Type u
              inst✝¹ : TopologicalSpace α
              R : Type u_2
              inst✝ : SeminormedRing R
              f : BoundedContinuousFunction α R
              ⊢ Eq (⇑(npowRec 0 f)) (HPow.hPow (⇑f) 0)
            -/
  | 0 => by rw [npowRec, pow_zero, coe_one]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u
                  inst✝¹ : TopologicalSpace α
                  R : Type u_2
                  inst✝ : SeminormedRing R
                  f : BoundedContinuousFunction α R
                  n : Nat
                  ⊢ Eq (⇑(npowRec (HAdd.hAdd n 1) f)) (HPow.hPow (⇑f) (HAdd.hAdd n 1))
                -/
  | n + 1 => by rw [npowRec, pow_succ, coe_mul, coe_npowRec f n]
                /-
                  🎉 no goals
                -/


instance hasNatPow : Pow (α →ᵇ R) ℕ where
  pow f n :=
    { toContinuousMap := f.toContinuousMap ^ n
                         /-
                           F : Type u_1
                           α : Type u
                           β : Type v
                           γ : Type w
                           inst✝¹ : TopologicalSpace α
                           R : Type u_2
                           inst✝ : SeminormedRing R
                           f : BoundedContinuousFunction α R
                           n : Nat
                           ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ((HPow.hPow f.toContinuousMap  …
                         -/
      map_bounded' := by simpa [coe_npowRec] using (npowRec n f).map_bounded' }
                         /-
                           🎉 no goals
                         -/


instance : NatCast (α →ᵇ R) :=
  ⟨fun n => BoundedContinuousFunction.const _ n⟩


@[simp, norm_cast]
theorem coe_natCast (n : ℕ) : ((n : α →ᵇ R) : α → R) = n := rfl


@[simp, norm_cast]
theorem coe_ofNat (n : ℕ) [n.AtLeastTwo] :
    ((ofNat(n) : α →ᵇ R) : α → R) = ofNat(n) :=
  rfl


instance : IntCast (α →ᵇ R) :=
  ⟨fun n => BoundedContinuousFunction.const _ n⟩


@[simp, norm_cast]
theorem coe_intCast (n : ℤ) : ((n : α →ᵇ R) : α → R) = n := rfl


instance instRing : Ring (α →ᵇ R) :=
  DFunLike.coe_injective.ring _ coe_zero coe_one coe_add coe_mul coe_neg coe_sub
    (fun _ _ => coe_nsmul _ _) (fun _ _ => coe_zsmul _ _) (fun _ _ => coe_pow _ _) coe_natCast
    coe_intCast


instance instSeminormedRing : SeminormedRing (α →ᵇ R) where
  __ := instRing
  __ := instNonUnitalSeminormedRing


instance instNormedRing [NormedRing R] : NormedRing (α →ᵇ R) where
  __ := instRing
  __ := instNonUnitalNormedRing


instance instCommRing [SeminormedCommRing R] : CommRing (α →ᵇ R) where
  mul_comm _ _ := ext fun _ ↦ mul_comm _ _


instance instSeminormedCommRing [SeminormedCommRing R] : SeminormedCommRing (α →ᵇ R) where
  __ := instCommRing
  __ := instSeminormedAddCommGroup
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): Added proof for `norm_mul`
  norm_mul := norm_mul_le


instance instNormedCommRing [NormedCommRing R] : NormedCommRing (α →ᵇ R) where
  __ := instCommRing
  __ := instNormedAddCommGroup
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): Added proof for `norm_mul`
  norm_mul := norm_mul_le


instance [IsScalarTower 𝕜 β β] : IsScalarTower 𝕜 (α →ᵇ β) (α →ᵇ β) where
  smul_assoc _ _ _ := ext fun _ ↦ smul_mul_assoc ..


instance [SMulCommClass 𝕜 β β] : SMulCommClass 𝕜 (α →ᵇ β) (α →ᵇ β) where
  smul_comm _ _ _ := ext fun _ ↦ (mul_smul_comm ..).symm


instance [SMulCommClass 𝕜 β β] : SMulCommClass (α →ᵇ β) 𝕜 (α →ᵇ β) where
  smul_comm _ _ _ := ext fun _ ↦ mul_smul_comm ..


/-- `BoundedContinuousFunction.const` as a `RingHom`. -/
def C : 𝕜 →+* α →ᵇ γ where
  toFun := fun c : 𝕜 => const α ((algebraMap 𝕜 γ) c)
  map_one' := ext fun _ => (algebraMap 𝕜 γ).map_one
  map_mul' _ _ := ext fun _ => (algebraMap 𝕜 γ).map_mul _ _
  map_zero' := ext fun _ => (algebraMap 𝕜 γ).map_zero
  map_add' _ _ := ext fun _ => (algebraMap 𝕜 γ).map_add _ _


instance instAlgebra : Algebra 𝕜 (α →ᵇ γ) where
  toRingHom := C
  commutes' _ _ := ext fun _ ↦ Algebra.commutes' _ _
  smul_def' _ _ := ext fun _ ↦ Algebra.smul_def' _ _


@[simp]
theorem algebraMap_apply (k : 𝕜) (a : α) : algebraMap 𝕜 (α →ᵇ γ) k a = k • (1 : γ) := by
  /-
    α : Type u
    γ : Type w
    𝕜 : Type u_2
    inst✝³ : NormedField 𝕜
    inst✝² : TopologicalSpace α
    inst✝¹ : NormedRing γ
    inst✝ : NormedAlgebra 𝕜 γ
    k : 𝕜
    a : α
    ⊢ Eq (((algebraMap 𝕜 (BoundedContinuousFunction α γ)) k) a) (HSMul.hSMul k 1)
  -/
  simp only [Algebra.algebraMap_eq_smul_one, coe_smul, coe_one, Pi.one_apply]
  /-
    🎉 no goals
  -/


instance instNormedAlgebra : NormedAlgebra 𝕜 (α →ᵇ γ) where
  __ := instAlgebra
  __ := instNormedSpace


instance instSMul' : SMul (α →ᵇ 𝕜) (α →ᵇ β) where
  smul f g :=
    ofNormedAddCommGroup (fun x => f x • g x) (f.continuous.smul g.continuous) (‖f‖ * ‖g‖) fun x =>
      calc
        ‖f x • g x‖ ≤ ‖f x‖ * ‖g x‖ := norm_smul_le _ _
        _ ≤ ‖f‖ * ‖g‖ :=
          mul_le_mul (f.norm_coe_le_norm _) (g.norm_coe_le_norm _) (norm_nonneg _) (norm_nonneg _)


instance instModule' : Module (α →ᵇ 𝕜) (α →ᵇ β) :=
  Module.ofMinimalAxioms
      (fun c _ _ => ext fun a => smul_add (c a) _ _)
      (fun _ _ _ => ext fun _ => add_smul _ _ _)
      (fun _ _ _ => ext fun _ => mul_smul _ _ _)
      (fun f => ext fun x => one_smul 𝕜 (f x))

/- TODO: When `NormedModule` has been added to `Analysis.Normed.Module.Basic`, this
shows that the space of bounded continuous functions from `α` to `β` is naturally a normed
module over the algebra of bounded continuous functions from `α` to `𝕜`. -/

instance : BoundedSMul (α →ᵇ 𝕜) (α →ᵇ β) :=
  BoundedSMul.of_norm_smul_le fun _ _ =>
    norm_ofNormedAddCommGroup_le _ (mul_nonneg (norm_nonneg _) (norm_nonneg _)) _


theorem NNReal.upper_bound {α : Type*} [TopologicalSpace α] (f : α →ᵇ ℝ≥0) (x : α) :
    f x ≤ nndist f 0 := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α NNReal
    x : α
    ⊢ LE.le (f x) (NNDist.nndist f 0)
  -/
  have key : nndist (f x) ((0 : α →ᵇ ℝ≥0) x) ≤ nndist f 0 := @dist_coe_le_dist α ℝ≥0 _ _ f 0 x
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α NNReal
    x : α
    key : LE.le (NNDist.nndist (f x) (0 x)) (NNDist.nndist f 0)
    ⊢ LE.le (f x) (NNDist.nndist f 0)
  -/
  simp only [coe_zero, Pi.zero_apply] at key
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α NNReal
    x : α
    key : LE.le (NNDist.nndist (f x) 0) (NNDist.nndist f 0)
    ⊢ LE.le (f x) (NNDist.nndist f 0)
  -/
  rwa [NNReal.nndist_zero_eq_val' (f x)] at key
  /-
    🎉 no goals
  -/


instance instPartialOrder : PartialOrder (α →ᵇ β) :=
                                           /-
                                             F : Type u_1
                                             α : Type u
                                             β : Type v
                                             γ : Type w
                                             inst✝¹ : TopologicalSpace α
                                             inst✝ : NormedLatticeAddCommGroup β
                                             ⊢ Function.Injective fun f => f.toFun
                                           -/
  PartialOrder.lift (fun f => f.toFun) (by simp [Injective])
                                           /-
                                             🎉 no goals
                                           -/


instance instSup : Max (α →ᵇ β) where
  max f g :=
    { toFun := f ⊔ g
      continuous_toFun := f.continuous.sup g.continuous
      map_bounded' := by
        /-
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := Max.max ⇑f ⇑g, con …
        -/
        obtain ⟨C₁, hf⟩ := f.bounded
        /-
          case intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ : Real
          hf : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C₁
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := Max.max ⇑f ⇑g, con …
        -/
        obtain ⟨C₂, hg⟩ := g.bounded
        /-
          case intro.intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ : Real
          hf : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C₁
          C₂ : Real
          hg : ∀ (x y : α), LE.le (Dist.dist (g x) (g y)) C₂
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := Max.max ⇑f ⇑g, con …
        -/
        refine ⟨C₁ + C₂, fun x y ↦ ?_⟩
        /-
          case intro.intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ : Real
          hf : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C₁
          C₂ : Real
          hg : ∀ (x y : α), LE.le (Dist.dist (g x) (g y)) C₂
          x y : α
          ⊢ LE.le (Dist.dist ({ toFun := Max.max ⇑f ⇑g, continuous_toFun := ⋯ }.toFun x) …
        -/
        simp_rw [NormedAddCommGroup.dist_eq] at hf hg ⊢
        /-
          case intro.intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ C₂ : Real
          x y : α
          hf : ∀ (x y : α), LE.le (Norm.norm (HSub.hSub (f x) (f y))) C₁
          hg : ∀ (x y : α), LE.le (Norm.norm (HSub.hSub (g x) (g y))) C₂
          ⊢ LE.le (Norm.norm (HSub.hSub (Max.max (⇑f) (⇑g) x) (Max.max (⇑f) (⇑g) y))) (H …
        -/
        exact (norm_sup_sub_sup_le_add_norm _ _ _ _).trans (add_le_add (hf _ _) (hg _ _)) }
        /-
          🎉 no goals
        -/


instance instInf : Min (α →ᵇ β) where
  min f g :=
    { toFun := f ⊓ g
      continuous_toFun := f.continuous.inf g.continuous
      map_bounded' := by
        /-
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := Min.min ⇑f ⇑g, con …
        -/
        obtain ⟨C₁, hf⟩ := f.bounded
        /-
          case intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ : Real
          hf : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C₁
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := Min.min ⇑f ⇑g, con …
        -/
        obtain ⟨C₂, hg⟩ := g.bounded
        /-
          case intro.intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ : Real
          hf : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C₁
          C₂ : Real
          hg : ∀ (x y : α), LE.le (Dist.dist (g x) (g y)) C₂
          ⊢ Exists fun C => ∀ (x y : α), LE.le (Dist.dist ({ toFun := Min.min ⇑f ⇑g, con …
        -/
        refine ⟨C₁ + C₂, fun x y ↦ ?_⟩
        /-
          case intro.intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ : Real
          hf : ∀ (x y : α), LE.le (Dist.dist (f x) (f y)) C₁
          C₂ : Real
          hg : ∀ (x y : α), LE.le (Dist.dist (g x) (g y)) C₂
          x y : α
          ⊢ LE.le (Dist.dist ({ toFun := Min.min ⇑f ⇑g, continuous_toFun := ⋯ }.toFun x) …
        -/
        simp_rw [NormedAddCommGroup.dist_eq] at hf hg ⊢
        /-
          case intro.intro
          F : Type u_1
          α : Type u
          β : Type v
          γ : Type w
          inst✝¹ : TopologicalSpace α
          inst✝ : NormedLatticeAddCommGroup β
          f g : BoundedContinuousFunction α β
          C₁ C₂ : Real
          x y : α
          hf : ∀ (x y : α), LE.le (Norm.norm (HSub.hSub (f x) (f y))) C₁
          hg : ∀ (x y : α), LE.le (Norm.norm (HSub.hSub (g x) (g y))) C₂
          ⊢ LE.le (Norm.norm (HSub.hSub (Min.min (⇑f) (⇑g) x) (Min.min (⇑f) (⇑g) y))) (H …
        -/
        exact (norm_inf_sub_inf_le_add_norm _ _ _ _).trans (add_le_add (hf _ _) (hg _ _)) }
        /-
          🎉 no goals
        -/


@[simp, norm_cast] lemma coe_sup (f g : α →ᵇ β) : ⇑(f ⊔ g) = ⇑f ⊔ ⇑g := rfl


@[simp, norm_cast] lemma coe_inf (f g : α →ᵇ β) : ⇑(f ⊓ g) = ⇑f ⊓ ⇑g := rfl


instance instSemilatticeSup : SemilatticeSup (α →ᵇ β) :=
  DFunLike.coe_injective.semilatticeSup _ coe_sup


instance instSemilatticeInf : SemilatticeInf (α →ᵇ β) :=
  DFunLike.coe_injective.semilatticeInf _ coe_inf


instance instLattice : Lattice (α →ᵇ β) := DFunLike.coe_injective.lattice _ coe_sup coe_inf


@[simp, norm_cast] lemma coe_abs (f : α →ᵇ β) : ⇑|f| = |⇑f| := rfl


@[simp, norm_cast] lemma coe_posPart (f : α →ᵇ β) : ⇑f⁺ = (⇑f)⁺ := rfl

@[simp, norm_cast] lemma coe_negPart (f : α →ᵇ β) : ⇑f⁻ = (⇑f)⁻ := rfl


@[deprecated (since := "2024-02-21")] alias coeFn_sup := coe_sup

@[deprecated (since := "2024-02-21")] alias coeFn_abs := coe_abs


instance instNormedLatticeAddCommGroup : NormedLatticeAddCommGroup (α →ᵇ β) :=
  { instSeminormedAddCommGroup with
    add_le_add_left := by
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        ⊢ ∀ (a b : BoundedContinuousFunction α β), LE.le a b → ∀ (c : BoundedContinuou …
      -/
      intro f g h₁ h t
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        f g : BoundedContinuousFunction α β
        h₁ : LE.le f g
        h : BoundedContinuousFunction α β
        t : α
        ⊢ LE.le ((fun f => ⇑f) (HAdd.hAdd h f) t) ((fun f => ⇑f) (HAdd.hAdd h g) t)
      -/
      simp only [coe_toContinuousMap, Pi.add_apply, add_le_add_iff_left, coe_add]
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        ⊢ ∀ ⦃x y : BoundedContinuousFunction α β⦄, LE.le (abs x) (abs y) → LE.le (Norm …
      -/
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        f g : BoundedContinuousFunction α β
        h₁ : LE.le f g
        h : BoundedContinuousFunction α β
        t : α
        ⊢ LE.le (f t) (g t)
      -/
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        f g : BoundedContinuousFunction α β
        h : LE.le (abs f) (abs g)
        ⊢ LE.le (Norm.norm f) (Norm.norm g)
      -/
      exact h₁ _
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        f g : BoundedContinuousFunction α β
        h : LE.le (abs f) (abs g)
        i1 : ∀ (t : α), LE.le (Norm.norm (f t)) (Norm.norm (g t))
        ⊢ LE.le (Norm.norm f) (Norm.norm g)
      -/
      /-
        🎉 no goals
      -/
      /-
        F : Type u_1
        α : Type u
        β : Type v
        γ : Type w
        inst✝¹ : TopologicalSpace α
        inst✝ : NormedLatticeAddCommGroup β
        f g : BoundedContinuousFunction α β
        h : LE.le (abs f) (abs g)
        i1 : ∀ (t : α), LE.le (Norm.norm (f t)) (Norm.norm (g t))
        ⊢ ∀ (x : α), LE.le (Norm.norm (f x)) (Norm.norm g)
      -/
    solid := by
      /-
        🎉 no goals
      -/
      intro f g h
      have i1 : ∀ t, ‖f t‖ ≤ ‖g t‖ := fun t => HasSolidNorm.solid (h t)
      rw [norm_le (norm_nonneg _)]
      exact fun t => (i1 t).trans (norm_coe_le_norm g t)
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10888): added proof for `eq_of_dist_eq_zero`
    eq_of_dist_eq_zero }


/-- The nonnegative part of a bounded continuous `ℝ`-valued function as a bounded
continuous `ℝ≥0`-valued function. -/
def nnrealPart (f : α →ᵇ ℝ) : α →ᵇ ℝ≥0 :=
  BoundedContinuousFunction.comp _ (show LipschitzWith 1 Real.toNNReal from lipschitzWith_posPart) f


@[simp]
theorem nnrealPart_coeFn_eq (f : α →ᵇ ℝ) : ⇑f.nnrealPart = Real.toNNReal ∘ ⇑f := rfl


/-- The absolute value of a bounded continuous `ℝ`-valued function as a bounded
continuous `ℝ≥0`-valued function. -/
def nnnorm (f : α →ᵇ ℝ) : α →ᵇ ℝ≥0 :=
  BoundedContinuousFunction.comp _
    (show LipschitzWith 1 fun x : ℝ => ‖x‖₊ from lipschitzWith_one_norm) f


@[simp]
theorem nnnorm_coeFn_eq (f : α →ᵇ ℝ) : ⇑f.nnnorm = NNNorm.nnnorm ∘ ⇑f := rfl

-- TODO: Use `posPart` and `negPart` here

/-- Decompose a bounded continuous function to its positive and negative parts. -/
theorem self_eq_nnrealPart_sub_nnrealPart_neg (f : α →ᵇ ℝ) :
    ⇑f = (↑) ∘ f.nnrealPart - (↑) ∘ (-f).nnrealPart := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    ⊢ Eq (⇑f) (HSub.hSub (Function.comp NNReal.toReal ⇑f.nnrealPart) (Function.com …
  -/
  funext x
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    x : α
    ⊢ Eq (f x) (HSub.hSub (Function.comp NNReal.toReal ⇑f.nnrealPart) (Function.co …
  -/
  dsimp
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    x : α
    ⊢ Eq (f x) (HSub.hSub (Max.max (f x) 0) (Max.max (Neg.neg (f x)) 0))
  -/
  simp only [max_zero_sub_max_neg_zero_eq_self]
  /-
    🎉 no goals
  -/


/-- Express the absolute value of a bounded continuous function in terms of its
positive and negative parts. -/
theorem abs_self_eq_nnrealPart_add_nnrealPart_neg (f : α →ᵇ ℝ) :
    abs ∘ ⇑f = (↑) ∘ f.nnrealPart + (↑) ∘ (-f).nnrealPart := by
  /-
    α : Type u
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    ⊢ Eq (Function.comp abs ⇑f) (HAdd.hAdd (Function.comp NNReal.toReal ⇑f.nnrealP …
  -/
  funext x
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    x : α
    ⊢ Eq (Function.comp abs (⇑f) x) (HAdd.hAdd (Function.comp NNReal.toReal ⇑f.nnr …
  -/
  dsimp
  /-
    case h
    α : Type u
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    x : α
    ⊢ Eq (abs (f x)) (HAdd.hAdd (Max.max (f x) 0) (Max.max (Neg.neg (f x)) 0))
  -/
  simp only [max_zero_add_max_neg_zero_eq_abs_self]
  /-
    🎉 no goals
  -/


lemma add_norm_nonneg (f : α →ᵇ ℝ) :
    0 ≤ f + const _ ‖f‖ := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    ⊢ LE.le 0 (HAdd.hAdd f (BoundedContinuousFunction.const α (Norm.norm f)))
  -/
  intro x
  simp only [ContinuousMap.toFun_eq_coe, coe_toContinuousMap, coe_zero, Pi.zero_apply, coe_add,
    const_apply, Pi.add_apply]
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    x : α
    ⊢ LE.le 0 (HAdd.hAdd (f x) (Norm.norm f))
  -/
  linarith [(abs_le.mp (norm_coe_le_norm f x)).1]
  /-
    🎉 no goals
  -/


lemma norm_sub_nonneg (f : α →ᵇ ℝ) :
    0 ≤ const _ ‖f‖ - f := by
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    ⊢ LE.le 0 (HSub.hSub (BoundedContinuousFunction.const α (Norm.norm f)) f)
  -/
  intro x
  simp only [ContinuousMap.toFun_eq_coe, coe_toContinuousMap, coe_zero, Pi.zero_apply, coe_sub,
    const_apply, Pi.sub_apply, sub_nonneg]
  /-
    α : Type u_2
    inst✝ : TopologicalSpace α
    f : BoundedContinuousFunction α Real
    x : α
    ⊢ LE.le (f x) (Norm.norm f)
  -/
  linarith [(abs_le.mp (norm_coe_le_norm f x)).2]
  /-
    🎉 no goals
  -/


