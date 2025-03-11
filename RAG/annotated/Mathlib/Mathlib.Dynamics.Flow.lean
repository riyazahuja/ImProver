/-- A set `s ⊆ α` is invariant under `ϕ : τ → α → α` if
    `ϕ t s ⊆ s` for all `t` in `τ`. -/
def IsInvariant (ϕ : τ → α → α) (s : Set α) : Prop :=
  ∀ t, MapsTo (ϕ t) s s


theorem isInvariant_iff_image : IsInvariant ϕ s ↔ ∀ t, ϕ t '' s ⊆ s := by
  /-
    τ : Type u_1
    α : Type u_2
    ϕ : τ → α → α
    s : Set α
    ⊢ Iff (IsInvariant ϕ s) (∀ (t : τ), HasSubset.Subset (Set.image (ϕ t) s) s)
  -/
  simp_rw [IsInvariant, mapsTo']
  /-
    🎉 no goals
  -/


/-- A set `s ⊆ α` is forward-invariant under `ϕ : τ → α → α` if
    `ϕ t s ⊆ s` for all `t ≥ 0`. -/
def IsFwInvariant [Preorder τ] [Zero τ] (ϕ : τ → α → α) (s : Set α) : Prop :=
  ∀ ⦃t⦄, 0 ≤ t → MapsTo (ϕ t) s s


theorem IsInvariant.isFwInvariant [Preorder τ] [Zero τ] {ϕ : τ → α → α} {s : Set α}
    (h : IsInvariant ϕ s) : IsFwInvariant ϕ s := fun t _ht => h t


/-- If `τ` is a `CanonicallyOrderedAddCommMonoid` (e.g., `ℕ` or `ℝ≥0`), then the notions
`IsFwInvariant` and `IsInvariant` are equivalent. -/
theorem IsFwInvariant.isInvariant [CanonicallyOrderedAddCommMonoid τ] {ϕ : τ → α → α} {s : Set α}
    (h : IsFwInvariant ϕ s) : IsInvariant ϕ s := fun t => h (zero_le t)


/-- If `τ` is a `CanonicallyOrderedAddCommMonoid` (e.g., `ℕ` or `ℝ≥0`), then the notions
`IsFwInvariant` and `IsInvariant` are equivalent. -/
theorem isFwInvariant_iff_isInvariant [CanonicallyOrderedAddCommMonoid τ] {ϕ : τ → α → α}
    {s : Set α} :
    IsFwInvariant ϕ s ↔ IsInvariant ϕ s :=
  ⟨IsFwInvariant.isInvariant, IsInvariant.isFwInvariant⟩


/-- A flow on a topological space `α` by an additive topological
    monoid `τ` is a continuous monoid action of `τ` on `α`. -/
structure Flow (τ : Type*) [TopologicalSpace τ] [AddMonoid τ] [ContinuousAdd τ] (α : Type*)
  [TopologicalSpace α] where
  /-- The map `τ → α → α` underlying a flow of `τ` on `α`. -/
  toFun : τ → α → α
  cont' : Continuous (uncurry toFun)
  map_add' : ∀ t₁ t₂ x, toFun (t₁ + t₂) x = toFun t₁ (toFun t₂ x)
  map_zero' : ∀ x, toFun 0 x = x


instance : Inhabited (Flow τ α) :=
  ⟨{  toFun := fun _ x => x
      cont' := continuous_snd
      map_add' := fun _ _ _ => rfl
      map_zero' := fun _ => rfl }⟩


instance : CoeFun (Flow τ α) fun _ => τ → α → α := ⟨Flow.toFun⟩


@[ext]
theorem ext : ∀ {ϕ₁ ϕ₂ : Flow τ α}, (∀ t x, ϕ₁ t x = ϕ₂ t x) → ϕ₁ = ϕ₂
  | ⟨f₁, _, _, _⟩, ⟨f₂, _, _, _⟩, h => by
    /-
      τ : Type u_1
      inst✝³ : AddMonoid τ
      inst✝² : TopologicalSpace τ
      inst✝¹ : ContinuousAdd τ
      α : Type u_2
      inst✝ : TopologicalSpace α
      f₁ : τ → α → α
      cont'✝¹ : Continuous (Function.uncurry f₁)
      map_add'✝¹ : ∀ (t₁ t₂ : τ) (x : α), Eq (f₁ (HAdd.hAdd t₁ t₂) x) (f₁ t₁ (f₁ t₂  …
      map_zero'✝¹ : ∀ (x : α), Eq (f₁ 0 x) x
      f₂ : τ → α → α
      cont'✝ : Continuous (Function.uncurry f₂)
      map_add'✝ : ∀ (t₁ t₂ : τ) (x : α), Eq (f₂ (HAdd.hAdd t₁ t₂) x) (f₂ t₁ (f₂ t₂ x))
      map_zero'✝ : ∀ (x : α), Eq (f₂ 0 x) x
      h : ∀ (t : τ) (x : α), Eq ({ toFun := f₁, cont' := cont'✝¹, map_add' := map_ad …
      ⊢ Eq { toFun := f₁, cont' := cont'✝¹, map_add' := map_add'✝¹, map_zero' := map …
    -/
    congr
    /-
      case e_toFun
      τ : Type u_1
      inst✝³ : AddMonoid τ
      inst✝² : TopologicalSpace τ
      inst✝¹ : ContinuousAdd τ
      α : Type u_2
      inst✝ : TopologicalSpace α
      f₁ : τ → α → α
      cont'✝¹ : Continuous (Function.uncurry f₁)
      map_add'✝¹ : ∀ (t₁ t₂ : τ) (x : α), Eq (f₁ (HAdd.hAdd t₁ t₂) x) (f₁ t₁ (f₁ t₂  …
      map_zero'✝¹ : ∀ (x : α), Eq (f₁ 0 x) x
      f₂ : τ → α → α
      cont'✝ : Continuous (Function.uncurry f₂)
      map_add'✝ : ∀ (t₁ t₂ : τ) (x : α), Eq (f₂ (HAdd.hAdd t₁ t₂) x) (f₂ t₁ (f₂ t₂ x))
      map_zero'✝ : ∀ (x : α), Eq (f₂ 0 x) x
      h : ∀ (t : τ) (x : α), Eq ({ toFun := f₁, cont' := cont'✝¹, map_add' := map_ad …
      ⊢ Eq f₁ f₂
    -/
    funext
    /-
      case e_toFun.h.h
      τ : Type u_1
      inst✝³ : AddMonoid τ
      inst✝² : TopologicalSpace τ
      inst✝¹ : ContinuousAdd τ
      α : Type u_2
      inst✝ : TopologicalSpace α
      f₁ : τ → α → α
      cont'✝¹ : Continuous (Function.uncurry f₁)
      map_add'✝¹ : ∀ (t₁ t₂ : τ) (x : α), Eq (f₁ (HAdd.hAdd t₁ t₂) x) (f₁ t₁ (f₁ t₂  …
      map_zero'✝¹ : ∀ (x : α), Eq (f₁ 0 x) x
      f₂ : τ → α → α
      cont'✝ : Continuous (Function.uncurry f₂)
      map_add'✝ : ∀ (t₁ t₂ : τ) (x : α), Eq (f₂ (HAdd.hAdd t₁ t₂) x) (f₂ t₁ (f₂ t₂ x))
      map_zero'✝ : ∀ (x : α), Eq (f₂ 0 x) x
      h : ∀ (t : τ) (x : α), Eq ({ toFun := f₁, cont' := cont'✝¹, map_add' := map_ad …
      x✝¹ : τ
      x✝ : α
      ⊢ Eq (f₁ x✝¹ x✝) (f₂ x✝¹ x✝)
    -/
    exact h _ _
    /-
      🎉 no goals
    -/


@[continuity, fun_prop]
protected theorem continuous {β : Type*} [TopologicalSpace β] {t : β → τ} (ht : Continuous t)
    {f : β → α} (hf : Continuous f) : Continuous fun x => ϕ (t x) (f x) :=
  ϕ.cont'.comp (ht.prod_mk hf)


alias _root_.Continuous.flow := Flow.continuous


theorem map_add (t₁ t₂ : τ) (x : α) : ϕ (t₁ + t₂) x = ϕ t₁ (ϕ t₂ x) := ϕ.map_add' _ _ _


@[simp]
theorem map_zero : ϕ 0 = id := funext ϕ.map_zero'


theorem map_zero_apply (x : α) : ϕ 0 x = x := ϕ.map_zero' x


/-- Iterations of a continuous function from a topological space `α`
    to itself defines a semiflow by `ℕ` on `α`. -/
def fromIter {g : α → α} (h : Continuous g) : Flow ℕ α where
  toFun n x := g^[n] x
  cont' := continuous_prod_of_discrete_left.mpr (Continuous.iterate h)
  map_add' := iterate_add_apply _
  map_zero' _x := rfl


/-- Restriction of a flow onto an invariant set. -/
def restrict {s : Set α} (h : IsInvariant ϕ s) : Flow τ (↥s) where
  toFun t := (h t).restrict _ _ _
  cont' := (ϕ.continuous continuous_fst continuous_subtype_val.snd').subtype_mk _
  map_add' _ _ _ := Subtype.ext (map_add _ _ _ _)
  map_zero' _ := Subtype.ext (map_zero_apply _ _)


theorem isInvariant_iff_image_eq (s : Set α) : IsInvariant ϕ s ↔ ∀ t, ϕ t '' s = s :=
  (isInvariant_iff_image _ _).trans
    (Iff.intro
                                                                                /-
                                                                                  τ : Type u_1
                                                                                  inst✝³ : AddCommGroup τ
                                                                                  inst✝² : TopologicalSpace τ
                                                                                  inst✝¹ : TopologicalAddGroup τ
                                                                                  α : Type u_2
                                                                                  inst✝ : TopologicalSpace α
                                                                                  ϕ : Flow τ α
                                                                                  s : Set α
                                                                                  h : ∀ (t : τ), HasSubset.Subset (Set.image (ϕ.toFun t) s) s
                                                                                  t : τ
                                                                                  x✝ : α
                                                                                  hx : Membership.mem s x✝
                                                                                  ⊢ Eq (ϕ.toFun t (ϕ.toFun (Neg.neg t) x✝)) x✝
                                                                                -/
      (fun h t => Subset.antisymm (h t) fun _ hx => ⟨_, h (-t) ⟨_, hx, rfl⟩, by simp [← map_add]⟩)
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                    /-
                      τ : Type u_1
                      inst✝³ : AddCommGroup τ
                      inst✝² : TopologicalSpace τ
                      inst✝¹ : TopologicalAddGroup τ
                      α : Type u_2
                      inst✝ : TopologicalSpace α
                      ϕ : Flow τ α
                      s : Set α
                      h : ∀ (t : τ), Eq (Set.image (ϕ.toFun t) s) s
                      t : τ
                      ⊢ HasSubset.Subset (Set.image (ϕ.toFun t) s) s
                    -/
      fun h t => by rw [h t])
                    /-
                      🎉 no goals
                    -/


/-- The time-reversal of a flow `ϕ` by a (commutative, additive) group
    is defined `ϕ.reverse t x = ϕ (-t) x`. -/
def reverse : Flow τ α where
  toFun t := ϕ (-t)
  cont' := ϕ.continuous continuous_fst.neg continuous_snd
                       /-
                         τ : Type u_1
                         inst✝³ : AddCommGroup τ
                         inst✝² : TopologicalSpace τ
                         inst✝¹ : TopologicalAddGroup τ
                         α : Type u_2
                         inst✝ : TopologicalSpace α
                         ϕ : Flow τ α
                         x✝² x✝¹ : τ
                         x✝ : α
                         ⊢ Eq ((fun t => ϕ.toFun (Neg.neg t)) (HAdd.hAdd x✝² x✝¹) x✝) ((fun t => ϕ.toFu …
                       -/
  map_add' _ _ _ := by dsimp; rw [neg_add, map_add]
                              /-
                                🎉 no goals
                              -/
                    /-
                      τ : Type u_1
                      inst✝³ : AddCommGroup τ
                      inst✝² : TopologicalSpace τ
                      inst✝¹ : TopologicalAddGroup τ
                      α : Type u_2
                      inst✝ : TopologicalSpace α
                      ϕ : Flow τ α
                      x✝ : α
                      ⊢ Eq ((fun t => ϕ.toFun (Neg.neg t)) 0 x✝) x✝
                    -/
  map_zero' _ := by dsimp; rw [neg_zero, map_zero_apply]
                           /-
                             🎉 no goals
                           -/

-- Porting note: add @continuity to Flow.toFun so that these works:
-- Porting note: Homeomorphism.continuous_toFun  : Continuous toFun  := by continuity
-- Porting note: Homeomorphism.continuous_invFun : Continuous invFun := by continuity

@[continuity]
theorem continuous_toFun (t : τ) : Continuous (ϕ.toFun t) := by
  /-
    τ : Type u_1
    inst✝³ : AddCommGroup τ
    inst✝² : TopologicalSpace τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    ϕ : Flow τ α
    t : τ
    ⊢ Continuous (ϕ.toFun t)
  -/
  rw [← curry_uncurry ϕ.toFun]
  /-
    τ : Type u_1
    inst✝³ : AddCommGroup τ
    inst✝² : TopologicalSpace τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    ϕ : Flow τ α
    t : τ
    ⊢ Continuous (Function.curry (Function.uncurry ϕ.toFun) t)
  -/
  apply continuous_curry
  /-
    case h
    τ : Type u_1
    inst✝³ : AddCommGroup τ
    inst✝² : TopologicalSpace τ
    inst✝¹ : TopologicalAddGroup τ
    α : Type u_2
    inst✝ : TopologicalSpace α
    ϕ : Flow τ α
    t : τ
    ⊢ Continuous (Function.uncurry ϕ.toFun)
  -/
  exact ϕ.cont'
  /-
    🎉 no goals
  -/


/-- The map `ϕ t` as a homeomorphism. -/
def toHomeomorph (t : τ) : (α ≃ₜ α) where
  toFun := ϕ t
  invFun := ϕ (-t)
                   /-
                     τ : Type u_1
                     inst✝³ : AddCommGroup τ
                     inst✝² : TopologicalSpace τ
                     inst✝¹ : TopologicalAddGroup τ
                     α : Type u_2
                     inst✝ : TopologicalSpace α
                     ϕ : Flow τ α
                     t : τ
                     x : α
                     ⊢ Eq (ϕ.toFun (Neg.neg t) (ϕ.toFun t x)) x
                   -/
  left_inv x := by rw [← map_add, neg_add_cancel, map_zero_apply]
                   /-
                     🎉 no goals
                   -/
                    /-
                      τ : Type u_1
                      inst✝³ : AddCommGroup τ
                      inst✝² : TopologicalSpace τ
                      inst✝¹ : TopologicalAddGroup τ
                      α : Type u_2
                      inst✝ : TopologicalSpace α
                      ϕ : Flow τ α
                      t : τ
                      x : α
                      ⊢ Eq (ϕ.toFun t (ϕ.toFun (Neg.neg t) x)) x
                    -/
  right_inv x := by rw [← map_add, add_neg_cancel, map_zero_apply]
                    /-
                      🎉 no goals
                    -/


theorem image_eq_preimage (t : τ) (s : Set α) : ϕ t '' s = ϕ (-t) ⁻¹' s :=
  (ϕ.toHomeomorph t).toEquiv.image_eq_preimage s


