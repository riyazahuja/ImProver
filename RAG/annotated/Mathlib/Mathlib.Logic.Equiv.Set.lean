@[simp]
theorem range_eq_univ {α : Type*} {β : Type*} (e : α ≃ β) : range e = univ :=
  eq_univ_of_forall e.surjective


protected theorem image_eq_preimage {α β} (e : α ≃ β) (s : Set α) : e '' s = e.symm ⁻¹' s :=
  Set.ext fun _ => mem_image_iff_of_inverse e.left_inv e.right_inv


@[simp 1001]
theorem _root_.Set.mem_image_equiv {α β} {S : Set α} {f : α ≃ β} {x : β} :
    x ∈ f '' S ↔ f.symm x ∈ S :=
  Set.ext_iff.mp (f.image_eq_preimage S) x


/-- Alias for `Equiv.image_eq_preimage` -/
theorem _root_.Set.image_equiv_eq_preimage_symm {α β} (S : Set α) (f : α ≃ β) :
    f '' S = f.symm ⁻¹' S :=
  f.image_eq_preimage S


/-- Alias for `Equiv.image_eq_preimage` -/
theorem _root_.Set.preimage_equiv_eq_image_symm {α β} (S : Set α) (f : β ≃ α) :
    f ⁻¹' S = f.symm '' S :=
  (f.symm.image_eq_preimage S).symm

-- Porting note: increased priority so this fires before `image_subset_iff`

@[simp high]
protected theorem symm_image_subset {α β} (e : α ≃ β) (s : Set α) (t : Set β) :
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         e : Equiv α β
                                         s : Set α
                                         t : Set β
                                         ⊢ Iff (HasSubset.Subset (Set.image (⇑e.symm) t) s) (HasSubset.Subset t (Set.im …
                                       -/
    e.symm '' t ⊆ s ↔ t ⊆ e '' s := by rw [image_subset_iff, e.image_eq_preimage]
                                       /-
                                         🎉 no goals
                                       -/

-- Porting note: increased priority so this fires before `image_subset_iff`

@[simp high]
protected theorem subset_symm_image {α β} (e : α ≃ β) (s : Set α) (t : Set β) :
    s ⊆ e.symm '' t ↔ e '' s ⊆ t :=
  calc
                                                 /-
                                                   α : Type u_1
                                                   β : Type u_2
                                                   e : Equiv α β
                                                   s : Set α
                                                   t : Set β
                                                   ⊢ Iff (HasSubset.Subset s (Set.image (⇑e.symm) t)) (HasSubset.Subset (Set.imag …
                                                 -/
    s ⊆ e.symm '' t ↔ e.symm.symm '' s ⊆ t := by rw [e.symm.symm_image_subset]
                                                 /-
                                                   🎉 no goals
                                                 -/
                         /-
                           α : Type u_1
                           β : Type u_2
                           e : Equiv α β
                           s : Set α
                           t : Set β
                           ⊢ Iff (HasSubset.Subset (Set.image (⇑e.symm.symm) s) t) (HasSubset.Subset (Set …
                         -/
    _ ↔ e '' s ⊆ t := by rw [e.symm_symm]
                         /-
                           🎉 no goals
                         -/


@[simp]
theorem symm_image_image {α β} (e : α ≃ β) (s : Set α) : e.symm '' (e '' s) = s :=
  e.leftInverse_symm.image_image s


theorem eq_image_iff_symm_image_eq {α β} (e : α ≃ β) (s : Set α) (t : Set β) :
    t = e '' s ↔ e.symm '' t = s :=
  (e.symm.injective.image_injective.eq_iff' (e.symm_image_image s)).symm


@[simp]
theorem image_symm_image {α β} (e : α ≃ β) (s : Set β) : e '' (e.symm '' s) = s :=
  e.symm.symm_image_image s


@[simp]
theorem image_preimage {α β} (e : α ≃ β) (s : Set β) : e '' (e ⁻¹' s) = s :=
  e.surjective.image_preimage s


@[simp]
theorem preimage_image {α β} (e : α ≃ β) (s : Set α) : e ⁻¹' (e '' s) = s :=
  e.injective.preimage_image s


protected theorem image_compl {α β} (f : Equiv α β) (s : Set α) : f '' sᶜ = (f '' s)ᶜ :=
  image_compl_eq f.bijective


@[simp]
theorem symm_preimage_preimage {α β} (e : α ≃ β) (s : Set β) : e.symm ⁻¹' (e ⁻¹' s) = s :=
  e.rightInverse_symm.preimage_preimage s


@[simp]
theorem preimage_symm_preimage {α β} (e : α ≃ β) (s : Set α) : e ⁻¹' (e.symm ⁻¹' s) = s :=
  e.leftInverse_symm.preimage_preimage s


theorem preimage_subset {α β} (e : α ≃ β) (s t : Set β) : e ⁻¹' s ⊆ e ⁻¹' t ↔ s ⊆ t :=
  e.surjective.preimage_subset_preimage_iff


theorem image_subset {α β} (e : α ≃ β) (s t : Set α) : e '' s ⊆ e '' t ↔ s ⊆ t :=
  image_subset_image_iff e.injective


@[simp]
theorem image_eq_iff_eq {α β} (e : α ≃ β) (s t : Set α) : e '' s = e '' t ↔ s = t :=
  image_eq_image e.injective


theorem preimage_eq_iff_eq_image {α β} (e : α ≃ β) (s t) : e ⁻¹' s = t ↔ s = e '' t :=
  Set.preimage_eq_iff_eq_image e.bijective


theorem eq_preimage_iff_image_eq {α β} (e : α ≃ β) (s t) : s = e ⁻¹' t ↔ e '' s = t :=
  Set.eq_preimage_iff_image_eq e.bijective


lemma setOf_apply_symm_eq_image_setOf {α β} (e : α ≃ β) (p : α → Prop) :
    {b | p (e.symm b)} = e '' {a | p a} := by
  /-
    α : Type u_1
    β : Type u_2
    e : Equiv α β
    p : α → Prop
    ⊢ Eq (setOf fun b => p (e.symm b)) (Set.image (⇑e) (setOf fun a => p a))
  -/
  rw [Equiv.image_eq_preimage, preimage_setOf_eq]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_assoc_preimage {α β γ} {s : Set α} {t : Set β} {u : Set γ} :
    Equiv.prodAssoc α β γ ⁻¹' s ×ˢ t ×ˢ u = (s ×ˢ t) ×ˢ u := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    u : Set γ
    ⊢ Eq (Set.preimage (⇑(Equiv.prodAssoc α β γ)) (SProd.sprod s (SProd.sprod t u) …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    u : Set γ
    x✝ : Prod (Prod α β) γ
    ⊢ Iff (Membership.mem (Set.preimage (⇑(Equiv.prodAssoc α β γ)) (SProd.sprod s  …
  -/
  simp [and_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_assoc_symm_preimage {α β γ} {s : Set α} {t : Set β} {u : Set γ} :
    (Equiv.prodAssoc α β γ).symm ⁻¹' (s ×ˢ t) ×ˢ u = s ×ˢ t ×ˢ u := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    u : Set γ
    ⊢ Eq (Set.preimage (⇑(Equiv.prodAssoc α β γ).symm) (SProd.sprod (SProd.sprod s …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    u : Set γ
    x✝ : Prod α (Prod β γ)
    ⊢ Iff (Membership.mem (Set.preimage (⇑(Equiv.prodAssoc α β γ).symm) (SProd.spr …
  -/
  simp [and_assoc]
  /-
    🎉 no goals
  -/

-- `@[simp]` doesn't like these lemmas, as it uses `Set.image_congr'` to turn `Equiv.prodAssoc`
-- into a lambda expression and then unfold it.

theorem prod_assoc_image {α β γ} {s : Set α} {t : Set β} {u : Set γ} :
    Equiv.prodAssoc α β γ '' (s ×ˢ t) ×ˢ u = s ×ˢ t ×ˢ u := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    u : Set γ
    ⊢ Eq (Set.image (⇑(Equiv.prodAssoc α β γ)) (SProd.sprod (SProd.sprod s t) u))  …
  -/
  simpa only [Equiv.image_eq_preimage] using prod_assoc_symm_preimage
  /-
    🎉 no goals
  -/


theorem prod_assoc_symm_image {α β γ} {s : Set α} {t : Set β} {u : Set γ} :
    (Equiv.prodAssoc α β γ).symm '' s ×ˢ t ×ˢ u = (s ×ˢ t) ×ˢ u := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    u : Set γ
    ⊢ Eq (Set.image (⇑(Equiv.prodAssoc α β γ).symm) (SProd.sprod s (SProd.sprod t  …
  -/
  simpa only [Equiv.image_eq_preimage] using prod_assoc_preimage
  /-
    🎉 no goals
  -/


/-- A set `s` in `α × β` is equivalent to the sigma-type `Σ x, {y | (x, y) ∈ s}`. -/
def setProdEquivSigma {α β : Type*} (s : Set (α × β)) :
    s ≃ Σx : α, { y : β | (x, y) ∈ s } where
                               /-
                                 α✝ : Sort u
                                 β✝ : Sort v
                                 γ : Sort w
                                 α : Type u_1
                                 β : Type u_2
                                 s : Set (Prod α β)
                                 x : ↑s
                                 ⊢ Membership.mem (setOf fun y => Membership.mem s { fst := (↑x).1, snd := y }) …
                               -/
  toFun x := ⟨x.1.1, x.1.2, by simp⟩
                               /-
                                 🎉 no goals
                               -/
  invFun x := ⟨(x.1, x.2.1), x.2.2⟩
  left_inv := fun ⟨⟨_, _⟩, _⟩ => rfl
  right_inv := fun ⟨_, _, _⟩ => rfl


/-- The subtypes corresponding to equal sets are equivalent. -/
@[simps! apply]
def setCongr {α : Type*} {s t : Set α} (h : s = t) : s ≃ t :=
  subtypeEquivProp h

-- We could construct this using `Equiv.Set.image e s e.injective`,
-- but this definition provides an explicit inverse.

/-- A set is equivalent to its image under an equivalence.
-/
@[simps]
def image {α β : Type*} (e : α ≃ β) (s : Set α) :
    s ≃ e '' s where
                        /-
                          α✝ : Sort u
                          β✝ : Sort v
                          γ : Sort w
                          α : Type u_1
                          β : Type u_2
                          e : Equiv α β
                          s : Set α
                          x : ↑s
                          ⊢ Membership.mem (Set.image (⇑e) s) (e ↑x)
                        -/
  toFun x := ⟨e x.1, by simp⟩
                        /-
                          🎉 no goals
                        -/
  invFun y :=
    ⟨e.symm y.1, by
      /-
        α✝ : Sort u
        β✝ : Sort v
        γ : Sort w
        α : Type u_1
        β : Type u_2
        e : Equiv α β
        s : Set α
        y : ↑(Set.image (⇑e) s)
        ⊢ Membership.mem s (e.symm ↑y)
      -/
      rcases y with ⟨-, ⟨a, ⟨m, rfl⟩⟩⟩
      /-
        case mk.intro.intro
        α✝ : Sort u
        β✝ : Sort v
        γ : Sort w
        α : Type u_1
        β : Type u_2
        e : Equiv α β
        s : Set α
        a : α
        m : Membership.mem s a
        ⊢ Membership.mem s (e.symm ↑⟨e a, ⋯⟩)
      -/
      simpa using m⟩
      /-
        🎉 no goals
      -/
                   /-
                     α✝ : Sort u
                     β✝ : Sort v
                     γ : Sort w
                     α : Type u_1
                     β : Type u_2
                     e : Equiv α β
                     s : Set α
                     x : ↑s
                     ⊢ Eq ((fun y => ⟨e.symm ↑y, ⋯⟩) ((fun x => ⟨e ↑x, ⋯⟩) x)) x
                   -/
  left_inv x := by simp
                   /-
                     🎉 no goals
                   -/
                    /-
                      α✝ : Sort u
                      β✝ : Sort v
                      γ : Sort w
                      α : Type u_1
                      β : Type u_2
                      e : Equiv α β
                      s : Set α
                      y : ↑(Set.image (⇑e) s)
                      ⊢ Eq ((fun x => ⟨e ↑x, ⋯⟩) ((fun y => ⟨e.symm ↑y, ⋯⟩) y)) y
                    -/
  right_inv y := by simp
                    /-
                      🎉 no goals
                    -/


/-- `univ α` is equivalent to `α`. -/
@[simps apply symm_apply]
protected def univ (α) : @univ α ≃ α :=
  ⟨Subtype.val, fun a => ⟨a, trivial⟩, fun ⟨_, _⟩ => rfl, fun _ => rfl⟩


/-- An empty set is equivalent to the `Empty` type. -/
protected def empty (α) : (∅ : Set α) ≃ Empty :=
  equivEmpty _


/-- An empty set is equivalent to a `PEmpty` type. -/
protected def pempty (α) : (∅ : Set α) ≃ PEmpty :=
  equivPEmpty _


/-- If sets `s` and `t` are separated by a decidable predicate, then `s ∪ t` is equivalent to
`s ⊕ t`. -/
protected def union' {α} {s t : Set α} (p : α → Prop) [DecidablePred p] (hs : ∀ x ∈ s, p x)
    (ht : ∀ x ∈ t, ¬p x) : (s ∪ t : Set α) ≃ s ⊕ t where
  toFun x :=
    if hp : p x then Sum.inl ⟨_, x.2.resolve_right fun xt => ht _ xt hp⟩
    else Sum.inr ⟨_, x.2.resolve_left fun xs => hp (hs _ xs)⟩
  invFun o :=
    match o with
    | Sum.inl x => ⟨x, Or.inl x.2⟩
    | Sum.inr x => ⟨x, Or.inr x.2⟩
                                /-
                                  α✝ : Sort u
                                  β : Sort v
                                  γ : Sort w
                                  α : Type ?u.12752
                                  s t : Set α
                                  p : α → Prop
                                  inst✝ : DecidablePred p
                                  hs : ∀ (x : α), Membership.mem s x → p x
                                  ht : ∀ (x : α), Membership.mem t x → Not (p x)
                                  x✝ : ↑(Union.union s t)
                                  x : α
                                  h' : Membership.mem (Union.union s t) x
                                  ⊢ Eq ((fun o => Equiv.Set.union'.match_1 (fun o => ↑(Union.union s t)) o (fun  …
                                -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  left_inv := fun ⟨x, h'⟩ => by by_cases h : p x <;> simp [h]
                                                     /-
                                                       🎉 no goals
                                                     -/
  right_inv o := by
    /-
      α✝ : Sort u
      β : Sort v
      γ : Sort w
      α : Type ?u.12752
      s t : Set α
      p : α → Prop
      inst✝ : DecidablePred p
      hs : ∀ (x : α), Membership.mem s x → p x
      ht : ∀ (x : α), Membership.mem t x → Not (p x)
      o : Sum ↑s ↑t
      ⊢ Eq ((fun x => dite (p ↑x) (fun hp => Sum.inl ⟨↑x, ⋯⟩) fun hp => Sum.inr ⟨↑x, …
    -/
    rcases o with (⟨x, h⟩ | ⟨x, h⟩) <;> [simp [hs _ h]; simp [ht _ h]]
    /-
      🎉 no goals
    -/


/-- If sets `s` and `t` are disjoint, then `s ∪ t` is equivalent to `s ⊕ t`. -/
protected def union {α} {s t : Set α} [DecidablePred fun x => x ∈ s] (H : Disjoint s t) :
    (s ∪ t : Set α) ≃ s ⊕ t :=
  Set.union' (fun x => x ∈ s) (fun _ => id) fun _ xt xs => Set.disjoint_left.mp H xs xt


theorem union_apply_left {α} {s t : Set α} [DecidablePred fun x => x ∈ s] (H : Disjoint s t)
    {a : (s ∪ t : Set α)} (ha : ↑a ∈ s) : Equiv.Set.union H a = Sum.inl ⟨a, ha⟩ :=
  dif_pos ha


theorem union_apply_right {α} {s t : Set α} [DecidablePred fun x => x ∈ s] (H : Disjoint s t)
    {a : (s ∪ t : Set α)} (ha : ↑a ∈ t) : Equiv.Set.union H a = Sum.inr ⟨a, ha⟩ :=
  dif_neg fun h => Set.disjoint_left.mp H h ha


@[simp]
theorem union_symm_apply_left {α} {s t : Set α} [DecidablePred fun x => x ∈ s] (H : Disjoint s t)
                                                            /-
                                                              α✝ : Sort u
                                                              β : Sort v
                                                              γ : Sort w
                                                              α : Type ?u.16346
                                                              s t : Set α
                                                              inst✝ : DecidablePred fun x => Membership.mem s x
                                                              H : Disjoint s t
                                                              a : ↑s
                                                              ⊢ Membership.mem (Union.union s t) ↑a
                                                            -/
    (a : s) : (Equiv.Set.union H).symm (Sum.inl a) = ⟨a, by simp⟩ :=
                                                            /-
                                                              🎉 no goals
                                                            -/
  rfl


@[simp]
theorem union_symm_apply_right {α} {s t : Set α} [DecidablePred fun x => x ∈ s] (H : Disjoint s t)
                                                            /-
                                                              α✝ : Sort u
                                                              β : Sort v
                                                              γ : Sort w
                                                              α : Type ?u.16929
                                                              s t : Set α
                                                              inst✝ : DecidablePred fun x => Membership.mem s x
                                                              H : Disjoint s t
                                                              a : ↑t
                                                              ⊢ Membership.mem (Union.union s t) ↑a
                                                            -/
    (a : t) : (Equiv.Set.union H).symm (Sum.inr a) = ⟨a, by simp⟩ :=
                                                            /-
                                                              🎉 no goals
                                                            -/
  rfl


/-- A singleton set is equivalent to a `PUnit` type. -/
protected def singleton {α} (a : α) : ({a} : Set α) ≃ PUnit.{u} :=
  ⟨fun _ => PUnit.unit, fun _ => ⟨a, mem_singleton _⟩, fun ⟨x, h⟩ => by
    /-
      α✝ : Sort u
      β : Sort v
      γ : Sort w
      α : Type ?u.17514
      a : α
      x✝ : ↑(Singleton.singleton a)
      x : α
      h : Membership.mem (Singleton.singleton a) x
      ⊢ Eq ((fun x => ⟨a, ⋯⟩) ((fun x => PUnit.unit) ⟨x, h⟩)) ⟨x, h⟩
    -/
    simp? at h says simp only [mem_singleton_iff] at h
    /-
      α✝ : Sort u
      β : Sort v
      γ : Sort w
      α : Type ?u.17514
      a : α
      x✝ : ↑(Singleton.singleton a)
      x : α
      h✝ : Membership.mem (Singleton.singleton a) x
      h : Eq x a
      ⊢ Eq ((fun x => ⟨a, ⋯⟩) ((fun x => PUnit.unit) ⟨x, h✝⟩)) ⟨x, h✝⟩
    -/
    subst x
    /-
      α✝ : Sort u
      β : Sort v
      γ : Sort w
      α : Type ?u.17514
      a : α
      x✝ : ↑(Singleton.singleton a)
      h : Eq a a
      ⊢ Eq ((fun x => ⟨a, ⋯⟩) ((fun x => PUnit.unit) ⟨a, ⋯⟩)) ⟨a, ⋯⟩
    -/
    rfl, fun ⟨⟩ => rfl⟩
    /-
      🎉 no goals
    -/


/-- Equal sets are equivalent.

TODO: this is the same as `Equiv.setCongr`! -/
@[simps! apply symm_apply]
protected def ofEq {α : Type u} {s t : Set α} (h : s = t) : s ≃ t :=
  Equiv.setCongr h


lemma Equiv.strictMono_setCongr {α : Type*} [PartialOrder α] {S T : Set α} (h : S = T) :
    StrictMono (setCongr h) := fun _ _ ↦ id


/-- If `a ∉ s`, then `insert a s` is equivalent to `s ⊕ PUnit`. -/
protected def insert {α} {s : Set.{u} α} [DecidablePred (· ∈ s)] {a : α} (H : a ∉ s) :
    (insert a s : Set α) ≃ s ⊕ PUnit.{u + 1} :=
  calc
                                                            /-
                                                              α✝ : Sort u
                                                              β : Sort v
                                                              γ : Sort w
                                                              α : Type u
                                                              s : Set α
                                                              inst✝ : DecidablePred fun x => Membership.mem s x
                                                              a : α
                                                              H : Not (Membership.mem s a)
                                                              ⊢ Eq (Insert.insert a s) (Union.union s (Singleton.singleton a))
                                                            -/
    (insert a s : Set α) ≃ ↥(s ∪ {a}) := Equiv.Set.ofEq (by simp)
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                   /-
                                                     α✝ : Sort u
                                                     β : Sort v
                                                     γ : Sort w
                                                     α : Type u
                                                     s : Set α
                                                     inst✝ : DecidablePred fun x => Membership.mem s x
                                                     a : α
                                                     H : Not (Membership.mem s a)
                                                     ⊢ Disjoint s (Singleton.singleton a)
                                                   -/
    _ ≃ s ⊕ ({a} : Set α) := Equiv.Set.union <| by simpa
                                                   /-
                                                     🎉 no goals
                                                   -/
    _ ≃ s ⊕ PUnit.{u + 1} := sumCongr (Equiv.refl _) (Equiv.Set.singleton _)


@[simp]
theorem insert_symm_apply_inl {α} {s : Set.{u} α} [DecidablePred (· ∈ s)] {a : α} (H : a ∉ s)
    (b : s) : (Equiv.Set.insert H).symm (Sum.inl b) = ⟨b, Or.inr b.2⟩ :=
  rfl


@[simp]
theorem insert_symm_apply_inr {α} {s : Set.{u} α} [DecidablePred (· ∈ s)] {a : α} (H : a ∉ s)
    (b : PUnit.{u + 1}) : (Equiv.Set.insert H).symm (Sum.inr b) = ⟨a, Or.inl rfl⟩ :=
  rfl


@[simp]
theorem insert_apply_left {α} {s : Set.{u} α} [DecidablePred (· ∈ s)] {a : α} (H : a ∉ s) :
    Equiv.Set.insert H ⟨a, Or.inl rfl⟩ = Sum.inr PUnit.unit :=
  (Equiv.Set.insert H).apply_eq_iff_eq_symm_apply.2 rfl


@[simp]
theorem insert_apply_right {α} {s : Set.{u} α} [DecidablePred (· ∈ s)] {a : α} (H : a ∉ s) (b : s) :
    Equiv.Set.insert H ⟨b, Or.inr b.2⟩ = Sum.inl b :=
  (Equiv.Set.insert H).apply_eq_iff_eq_symm_apply.2 rfl


/-- If `s : Set α` is a set with decidable membership, then `s ⊕ sᶜ` is equivalent to `α`. -/
protected def sumCompl {α} (s : Set α) [DecidablePred (· ∈ s)] : s ⊕ (sᶜ : Set α) ≃ α :=
  calc
    s ⊕ (sᶜ : Set α) ≃ ↥(s ∪ sᶜ) := (Equiv.Set.union disjoint_compl_right).symm
                                      /-
                                        α✝ : Sort u
                                        β : Sort v
                                        γ : Sort w
                                        α : Type ?u.21708
                                        s : Set α
                                        inst✝ : DecidablePred fun x => Membership.mem s x
                                        ⊢ Eq (Union.union s (HasCompl.compl s)) _root_.Set.univ
                                      -/
    _ ≃ @univ α := Equiv.Set.ofEq (by simp)
                                      /-
                                        🎉 no goals
                                      -/
    _ ≃ α := Equiv.Set.univ _


@[simp]
theorem sumCompl_apply_inl {α : Type u} (s : Set α) [DecidablePred (· ∈ s)] (x : s) :
    Equiv.Set.sumCompl s (Sum.inl x) = x :=
  rfl


@[simp]
theorem sumCompl_apply_inr {α : Type u} (s : Set α) [DecidablePred (· ∈ s)] (x : (sᶜ : Set α)) :
    Equiv.Set.sumCompl s (Sum.inr x) = x :=
  rfl


theorem sumCompl_symm_apply_of_mem {α : Type u} {s : Set α} [DecidablePred (· ∈ s)] {x : α}
    (hx : x ∈ s) : (Equiv.Set.sumCompl s).symm x = Sum.inl ⟨x, hx⟩ := by
  /-
    α : Type u
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : α
    hx : Membership.mem s x
    ⊢ Eq ((Equiv.Set.sumCompl s).symm x) (Sum.inl ⟨x, hx⟩)
  -/
  simp [Equiv.Set.sumCompl, Equiv.Set.univ, union_apply_left, hx]
  /-
    🎉 no goals
  -/


theorem sumCompl_symm_apply_of_not_mem {α : Type u} {s : Set α} [DecidablePred (· ∈ s)] {x : α}
    (hx : x ∉ s) : (Equiv.Set.sumCompl s).symm x = Sum.inr ⟨x, hx⟩ := by
  /-
    α : Type u
    s : Set α
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : α
    hx : Not (Membership.mem s x)
    ⊢ Eq ((Equiv.Set.sumCompl s).symm x) (Sum.inr ⟨x, hx⟩)
  -/
  simp [Equiv.Set.sumCompl, Equiv.Set.univ, union_apply_right, hx]
  /-
    🎉 no goals
  -/


@[simp]
theorem sumCompl_symm_apply {α : Type*} {s : Set α} [DecidablePred (· ∈ s)] {x : s} :
    (Equiv.Set.sumCompl s).symm x = Sum.inl x :=
  Set.sumCompl_symm_apply_of_mem x.2


@[simp]
theorem sumCompl_symm_apply_compl {α : Type*} {s : Set α} [DecidablePred (· ∈ s)]
    {x : (sᶜ : Set α)} : (Equiv.Set.sumCompl s).symm x = Sum.inr x :=
  Set.sumCompl_symm_apply_of_not_mem x.2


/-- `sumDiffSubset s t` is the natural equivalence between
`s ⊕ (t \ s)` and `t`, where `s` and `t` are two sets. -/
protected def sumDiffSubset {α} {s t : Set α} (h : s ⊆ t) [DecidablePred (· ∈ s)] :
    s ⊕ (t \ s : Set α) ≃ t :=
  calc
    s ⊕ (t \ s : Set α) ≃ (s ∪ t \ s : Set α) :=
      (Equiv.Set.union disjoint_sdiff_self_right).symm
                                /-
                                  α✝ : Sort u
                                  β : Sort v
                                  γ : Sort w
                                  α : Type ?u.26200
                                  s t : Set α
                                  h : HasSubset.Subset s t
                                  inst✝ : DecidablePred fun x => Membership.mem s x
                                  ⊢ Eq (Union.union s (SDiff.sdiff t s)) t
                                -/
    _ ≃ t := Equiv.Set.ofEq (by simp [union_diff_self, union_eq_self_of_subset_left h])
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem sumDiffSubset_apply_inl {α} {s t : Set α} (h : s ⊆ t) [DecidablePred (· ∈ s)] (x : s) :
    Equiv.Set.sumDiffSubset h (Sum.inl x) = inclusion h x :=
  rfl


@[simp]
theorem sumDiffSubset_apply_inr {α} {s t : Set α} (h : s ⊆ t) [DecidablePred (· ∈ s)]
    (x : (t \ s : Set α)) : Equiv.Set.sumDiffSubset h (Sum.inr x) = inclusion diff_subset x :=
  rfl


theorem sumDiffSubset_symm_apply_of_mem {α} {s t : Set α} (h : s ⊆ t) [DecidablePred (· ∈ s)]
    {x : t} (hx : x.1 ∈ s) : (Equiv.Set.sumDiffSubset h).symm x = Sum.inl ⟨x, hx⟩ := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : ↑t
    hx : Membership.mem s ↑x
    ⊢ Eq ((Equiv.Set.sumDiffSubset h).symm x) (Sum.inl ⟨↑x, hx⟩)
  -/
  apply (Equiv.Set.sumDiffSubset h).injective
  /-
    case a
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : ↑t
    hx : Membership.mem s ↑x
    ⊢ Eq ((Equiv.Set.sumDiffSubset h) ((Equiv.Set.sumDiffSubset h).symm x)) ((Equi …
  -/
  simp only [apply_symm_apply, sumDiffSubset_apply_inl, Set.inclusion_mk]
  /-
    🎉 no goals
  -/


theorem sumDiffSubset_symm_apply_of_not_mem {α} {s t : Set α} (h : s ⊆ t) [DecidablePred (· ∈ s)]
    {x : t} (hx : x.1 ∉ s) : (Equiv.Set.sumDiffSubset h).symm x = Sum.inr ⟨x, ⟨x.2, hx⟩⟩ := by
  /-
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : ↑t
    hx : Not (Membership.mem s ↑x)
    ⊢ Eq ((Equiv.Set.sumDiffSubset h).symm x) (Sum.inr ⟨↑x, ⋯⟩)
  -/
  apply (Equiv.Set.sumDiffSubset h).injective
  /-
    case a
    α : Type u_1
    s t : Set α
    h : HasSubset.Subset s t
    inst✝ : DecidablePred fun x => Membership.mem s x
    x : ↑t
    hx : Not (Membership.mem s ↑x)
    ⊢ Eq ((Equiv.Set.sumDiffSubset h) ((Equiv.Set.sumDiffSubset h).symm x)) ((Equi …
  -/
  simp only [apply_symm_apply, sumDiffSubset_apply_inr, Set.inclusion_mk]
  /-
    🎉 no goals
  -/


/-- If `s` is a set with decidable membership, then the sum of `s ∪ t` and `s ∩ t` is equivalent
to `s ⊕ t`. -/
protected def unionSumInter {α : Type u} (s t : Set α) [DecidablePred (· ∈ s)] :
    (s ∪ t : Set α) ⊕ (s ∩ t : Set α) ≃ s ⊕ t :=
  calc
    (s ∪ t : Set α) ⊕ (s ∩ t : Set α)
                                                    /-
                                                      α✝ : Sort u
                                                      β : Sort v
                                                      γ : Sort w
                                                      α : Type u
                                                      s t : Set α
                                                      inst✝ : DecidablePred fun x => Membership.mem s x
                                                      ⊢ Equiv (Sum ↑(Union.union s t) ↑(Inter.inter s t)) (Sum ↑(Union.union s (SDif …
                                                    -/
      ≃ (s ∪ t \ s : Set α) ⊕ (s ∩ t : Set α) := by rw [union_diff_self]
                                                    /-
                                                      🎉 no goals
                                                    -/
    _ ≃ (s ⊕ (t \ s : Set α)) ⊕ (s ∩ t : Set α) :=
      sumCongr (Set.union disjoint_sdiff_self_right) (Equiv.refl _)
    _ ≃ s ⊕ ((t \ s : Set α) ⊕ (s ∩ t : Set α)) := sumAssoc _ _ _
    _ ≃ s ⊕ (t \ s ∪ s ∩ t : Set α) :=
      sumCongr (Equiv.refl _)
        (by
          /-
            α✝ : Sort u
            β : Sort v
            γ : Sort w
            α : Type u
            s t : Set α
            inst✝ : DecidablePred fun x => Membership.mem s x
            ⊢ Equiv (Sum ↑(SDiff.sdiff t s) ↑(Inter.inter s t)) ↑(Union.union (SDiff.sdiff …
          -/
          refine (Set.union' (· ∉ s) ?_ ?_).symm
          /-
            case refine_1
            α✝ : Sort u
            β : Sort v
            γ : Sort w
            α : Type u
            s t : Set α
            inst✝ : DecidablePred fun x => Membership.mem s x
            ⊢ ∀ (x : α), Membership.mem (SDiff.sdiff t s) x → (fun x => Not (Membership.me …
          -/
          exacts [fun x hx => hx.2, fun x hx => not_not_intro hx.1])
          /-
            🎉 no goals
          -/
    _ ≃ s ⊕ t := by
      { rw [(_ : t \ s ∪ s ∩ t = t)]
        rw [union_comm, inter_comm, inter_union_diff] }


/-- Given an equivalence `e₀` between sets `s : Set α` and `t : Set β`, the set of equivalences
`e : α ≃ β` such that `e ↑x = ↑(e₀ x)` for each `x : s` is equivalent to the set of equivalences
between `sᶜ` and `tᶜ`. -/
protected def compl {α : Type u} {β : Type v} {s : Set α} {t : Set β} [DecidablePred (· ∈ s)]
    [DecidablePred (· ∈ t)] (e₀ : s ≃ t) :
    { e : α ≃ β // ∀ x : s, e x = e₀ x } ≃ ((sᶜ : Set α) ≃ (tᶜ : Set β)) where
  toFun e :=
    subtypeEquiv e fun _ =>
      not_congr <|
        Iff.symm <|
          MapsTo.mem_iff (mapsTo_iff_exists_map_subtype.2 ⟨e₀, e.2⟩)
            (SurjOn.mapsTo_compl
              (surjOn_iff_exists_map_subtype.2 ⟨t, e₀, Subset.refl t, e₀.surjective, e.2⟩)
              e.1.injective)
  invFun e₁ :=
    Subtype.mk
      (calc
        α ≃ s ⊕ (sᶜ : Set α) := (Set.sumCompl s).symm
        _ ≃ t ⊕ (tᶜ : Set β) := e₀.sumCongr e₁
        _ ≃ β := Set.sumCompl t
        )
      fun x => by
      simp only [Sum.map_inl, trans_apply, sumCongr_apply, Set.sumCompl_apply_inl,
        Set.sumCompl_symm_apply, Trans.trans]
  left_inv e := by
    /-
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u
      β : Type v
      s : Set α
      t : Set β
      inst✝¹ : DecidablePred fun x => Membership.mem s x
      inst✝ : DecidablePred fun x => Membership.mem t x
      e₀ : Equiv ↑s ↑t
      e : Subtype fun e => ∀ (x : ↑s), Eq (e ↑x) ↑(e₀ x)
      ⊢ Eq ((fun e₁ => ⟨Trans.trans (Trans.trans (Equiv.Set.sumCompl s).symm (e₀.sum …
    -/
    ext x
    /-
      case a.H
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u
      β : Type v
      s : Set α
      t : Set β
      inst✝¹ : DecidablePred fun x => Membership.mem s x
      inst✝ : DecidablePred fun x => Membership.mem t x
      e₀ : Equiv ↑s ↑t
      e : Subtype fun e => ∀ (x : ↑s), Eq (e ↑x) ↑(e₀ x)
      x : α
      ⊢ Eq (↑((fun e₁ => ⟨Trans.trans (Trans.trans (Equiv.Set.sumCompl s).symm (e₀.s …
    -/
    by_cases hx : x ∈ s
    · simp only [Set.sumCompl_symm_apply_of_mem hx, ← e.prop ⟨x, hx⟩, Sum.map_inl, sumCongr_apply,
        trans_apply, Subtype.coe_mk, Set.sumCompl_apply_inl, Trans.trans]
    · simp only [Set.sumCompl_symm_apply_of_not_mem hx, Sum.map_inr, subtypeEquiv_apply,
        Set.sumCompl_apply_inr, trans_apply, sumCongr_apply, Subtype.coe_mk, Trans.trans]
  right_inv e :=
    Equiv.ext fun x => by
      simp only [Sum.map_inr, subtypeEquiv_apply, Set.sumCompl_apply_inr, Function.comp_apply,
        sumCongr_apply, Equiv.coe_trans, Subtype.coe_eta, Subtype.coe_mk, Trans.trans,
        Set.sumCompl_symm_apply_compl]


/-- The set product of two sets is equivalent to the type product of their coercions to types. -/
protected def prod {α β} (s : Set α) (t : Set β) : ↥(s ×ˢ t) ≃ s × t :=
  @subtypeProdEquivProd α β s t


/-- The set `Set.pi Set.univ s` is equivalent to `Π a, s a`. -/
@[simps]
protected def univPi {α : Type*} {β : α → Type*} (s : ∀ a, Set (β a)) :
    pi univ s ≃ ∀ a, s a where
  toFun f a := ⟨(f : ∀ a, β a) a, f.2 a (mem_univ a)⟩
  invFun f := ⟨fun a => f a, fun a _ => (f a).2⟩
  left_inv := fun ⟨f, hf⟩ => by
    /-
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u_1
      β : α → Type u_2
      s : (a : α) → Set (β a)
      x✝ : ↑(_root_.Set.univ.pi s)
      f : (i : α) → β i
      hf : Membership.mem (_root_.Set.univ.pi s) f
      ⊢ Eq ((fun f => ⟨fun a => ↑(f a), ⋯⟩) ((fun f a => ⟨↑f a, ⋯⟩) ⟨f, hf⟩)) ⟨f, hf⟩
    -/
    ext a
    /-
      case a.h
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u_1
      β : α → Type u_2
      s : (a : α) → Set (β a)
      x✝ : ↑(_root_.Set.univ.pi s)
      f : (i : α) → β i
      hf : Membership.mem (_root_.Set.univ.pi s) f
      a : α
      ⊢ Eq (↑((fun f => ⟨fun a => ↑(f a), ⋯⟩) ((fun f a => ⟨↑f a, ⋯⟩) ⟨f, hf⟩)) a) ( …
    -/
    rfl
    /-
      🎉 no goals
    -/
  right_inv f := by
    /-
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u_1
      β : α → Type u_2
      s : (a : α) → Set (β a)
      f : (a : α) → ↑(s a)
      ⊢ Eq ((fun f a => ⟨↑f a, ⋯⟩) ((fun f => ⟨fun a => ↑(f a), ⋯⟩) f)) f
    -/
    ext a
    /-
      case h.a
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u_1
      β : α → Type u_2
      s : (a : α) → Set (β a)
      f : (a : α) → ↑(s a)
      a : α
      ⊢ Eq ↑((fun f a => ⟨↑f a, ⋯⟩) ((fun f => ⟨fun a => ↑(f a), ⋯⟩) f) a) ↑(f a)
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- If a function `f` is injective on a set `s`, then `s` is equivalent to `f '' s`. -/
protected noncomputable def imageOfInjOn {α β} (f : α → β) (s : Set α) (H : InjOn f s) :
    s ≃ f '' s :=
  ⟨fun p => ⟨f p, mem_image_of_mem f p.2⟩, fun p =>
    ⟨Classical.choose p.2, (Classical.choose_spec p.2).1⟩, fun ⟨_, h⟩ =>
    Subtype.eq
      (H (Classical.choose_spec (mem_image_of_mem f h)).1 h
        (Classical.choose_spec (mem_image_of_mem f h)).2),
    fun ⟨_, h⟩ => Subtype.eq (Classical.choose_spec h).2⟩


/-- If `f` is an injective function, then `s` is equivalent to `f '' s`. -/
@[simps! apply]
protected noncomputable def image {α β} (f : α → β) (s : Set α) (H : Injective f) : s ≃ f '' s :=
  Equiv.Set.imageOfInjOn f s H.injOn


@[simp]
protected theorem image_symm_apply {α β} (f : α → β) (s : Set α) (H : Injective f) (x : α)
    (h : f x ∈ f '' s) : (Set.image f s H).symm ⟨f x, h⟩ = ⟨x, H.mem_set_image.1 h⟩ :=
  (Equiv.symm_apply_eq _).2 rfl


theorem image_symm_preimage {α β} {f : α → β} (hf : Injective f) (u s : Set α) :
    (fun x => (Set.image f s hf).symm x : f '' s → α) ⁻¹' u = Subtype.val ⁻¹' (f '' u) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    u s : Set α
    ⊢ Eq (Set.preimage (fun x => ↑((Equiv.Set.image f s hf).symm x)) u) (Set.preim …
  -/
  ext ⟨b, a, has, rfl⟩
  /-
    case h.mk.intro.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    u s : Set α
    a : α
    has : Membership.mem s a
    ⊢ Iff (Membership.mem (Set.preimage (fun x => ↑((Equiv.Set.image f s hf).symm  …
  -/
  simp [hf.eq_iff]
  /-
    🎉 no goals
  -/


/-- If `α` is equivalent to `β`, then `Set α` is equivalent to `Set β`. -/
@[simps]
protected def congr {α β : Type*} (e : α ≃ β) : Set α ≃ Set β :=
  ⟨fun s => e '' s, fun t => e.symm '' t, symm_image_image e, symm_image_image e.symm⟩


/-- The set `{x ∈ s | t x}` is equivalent to the set of `x : s` such that `t x`. -/
protected def sep {α : Type u} (s : Set α) (t : α → Prop) :
    ({ x ∈ s | t x } : Set α) ≃ { x : s | t x } :=
  (Equiv.subtypeSubtypeEquivSubtypeInter s t).symm


/-- The set `𝒫 S := {x | x ⊆ S}` is equivalent to the type `Set S`. -/
protected def powerset {α} (S : Set α) :
    𝒫 S ≃ Set S where
  toFun := fun x : 𝒫 S => Subtype.val ⁻¹' (x : Set α)
                                                   /-
                                                     α✝ : Sort u
                                                     β : Sort v
                                                     γ : Sort w
                                                     α : Type ?u.43249
                                                     S : Set α
                                                     x : Set ↑S
                                                     ⊢ Membership.mem S.powerset (_root_.Set.image Subtype.val x)
                                                   -/
  invFun := fun x : Set S => ⟨Subtype.val '' x, by rintro _ ⟨a : S, _, rfl⟩; exact a.2⟩
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
                   /-
                     α✝ : Sort u
                     β : Sort v
                     γ : Sort w
                     α : Type ?u.43249
                     S : Set α
                     x : ↑S.powerset
                     ⊢ Eq ((fun x => ⟨_root_.Set.image Subtype.val x, ⋯⟩) ((fun x => Set.preimage S …
                   -/
  left_inv x := by ext y;exact ⟨fun ⟨⟨_, _⟩, h, rfl⟩ => h, fun h => ⟨⟨_, x.2 h⟩, h, rfl⟩⟩
                         /-
                           🎉 no goals
                         -/
                    /-
                      α✝ : Sort u
                      β : Sort v
                      γ : Sort w
                      α : Type ?u.43249
                      S : Set α
                      x : Set ↑S
                      ⊢ Eq ((fun x => Set.preimage Subtype.val ↑x) ((fun x => ⟨_root_.Set.image Subt …
                    -/
  right_inv x := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- If `s` is a set in `range f`,
then its image under `rangeSplitting f` is in bijection (via `f`) with `s`.
-/
@[simps]
noncomputable def rangeSplittingImageEquiv {α β : Type*} (f : α → β) (s : Set (range f)) :
    rangeSplitting f '' s ≃ s where
  toFun x :=
              /-
                α✝ : Sort u
                β✝ : Sort v
                γ : Sort w
                α : Type u_1
                β : Type u_2
                f : α → β
                s : Set ↑(Set.range f)
                x : ↑(_root_.Set.image (Set.rangeSplitting f) s)
                ⊢ Membership.mem (Set.range f) (f ↑x)
              -/
    ⟨⟨f x, by simp⟩, by
              /-
                🎉 no goals
              -/
      /-
        α✝ : Sort u
        β✝ : Sort v
        γ : Sort w
        α : Type u_1
        β : Type u_2
        f : α → β
        s : Set ↑(Set.range f)
        x : ↑(_root_.Set.image (Set.rangeSplitting f) s)
        ⊢ Membership.mem s ⟨f ↑x, ⋯⟩
      -/
      rcases x with ⟨x, ⟨y, ⟨m, rfl⟩⟩⟩
      /-
        case mk.intro.intro
        α✝ : Sort u
        β✝ : Sort v
        γ : Sort w
        α : Type u_1
        β : Type u_2
        f : α → β
        s : Set ↑(Set.range f)
        y : ↑(Set.range f)
        m : Membership.mem s y
        ⊢ Membership.mem s ⟨f ↑⟨Set.rangeSplitting f y, ⋯⟩, ⋯⟩
      -/
      simpa [apply_rangeSplitting f] using m⟩
      /-
        🎉 no goals
      -/
  invFun x := ⟨rangeSplitting f x, ⟨x, ⟨x.2, rfl⟩⟩⟩
  left_inv x := by
    /-
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set ↑(Set.range f)
      x : ↑(_root_.Set.image (Set.rangeSplitting f) s)
      ⊢ Eq ((fun x => ⟨Set.rangeSplitting f ↑x, ⋯⟩) ((fun x => ⟨⟨f ↑x, ⋯⟩, ⋯⟩) x)) x
    -/
    rcases x with ⟨x, ⟨y, ⟨m, rfl⟩⟩⟩
    /-
      case mk.intro.intro
      α✝ : Sort u
      β✝ : Sort v
      γ : Sort w
      α : Type u_1
      β : Type u_2
      f : α → β
      s : Set ↑(Set.range f)
      y : ↑(Set.range f)
      m : Membership.mem s y
      ⊢ Eq ((fun x => ⟨Set.rangeSplitting f ↑x, ⋯⟩) ((fun x => ⟨⟨f ↑x, ⋯⟩, ⋯⟩) ⟨Set. …
    -/
    simp [apply_rangeSplitting f]
    /-
      🎉 no goals
    -/
                    /-
                      α✝ : Sort u
                      β✝ : Sort v
                      γ : Sort w
                      α : Type u_1
                      β : Type u_2
                      f : α → β
                      s : Set ↑(Set.range f)
                      x : ↑s
                      ⊢ Eq ((fun x => ⟨⟨f ↑x, ⋯⟩, ⋯⟩) ((fun x => ⟨Set.rangeSplitting f ↑x, ⋯⟩) x)) x
                    -/
  right_inv x := by simp [apply_rangeSplitting f]
                    /-
                      🎉 no goals
                    -/


/-- Equivalence between the range of `Sum.inl : α → α ⊕ β` and `α`. -/
@[simps symm_apply_coe]
def rangeInl (α β : Type*) : Set.range (Sum.inl : α → α ⊕ β) ≃ α where
  toFun
  | ⟨.inl x, _⟩ => x
                                    /-
                                      α✝ : Sort u
                                      β✝ : Sort v
                                      γ : Sort w
                                      α : Type u_1
                                      β : Type u_2
                                      val✝ : β
                                      h : Membership.mem (Set.range Sum.inl) (Sum.inr val✝)
                                      ⊢ False
                                    -/
  | ⟨.inr _, h⟩ => False.elim <| by rcases h with ⟨x, h'⟩; cases h'
                                                           /-
                                                             🎉 no goals
                                                           -/
  invFun x := ⟨.inl x, mem_range_self _⟩
  left_inv := fun ⟨_, _, rfl⟩ => rfl
  right_inv _ := rfl


@[simp] lemma rangeInl_apply_inl {α : Type*} (β : Type*) (x : α) :
    (rangeInl α β) ⟨.inl x, mem_range_self _⟩ = x :=
  rfl


/-- Equivalence between the range of `Sum.inr : β → α ⊕ β` and `β`. -/
@[simps symm_apply_coe]
def rangeInr (α β : Type*) : Set.range (Sum.inr : β → α ⊕ β) ≃ β where
  toFun
                                    /-
                                      α✝ : Sort u
                                      β✝ : Sort v
                                      γ : Sort w
                                      α : Type u_1
                                      β : Type u_2
                                      val✝ : α
                                      h : Membership.mem (Set.range Sum.inr) (Sum.inl val✝)
                                      ⊢ False
                                    -/
  | ⟨.inl _, h⟩ => False.elim <| by rcases h with ⟨x, h'⟩; cases h'
                                                           /-
                                                             🎉 no goals
                                                           -/
  | ⟨.inr x, _⟩ => x
  invFun x := ⟨.inr x, mem_range_self _⟩
  left_inv := fun ⟨_, _, rfl⟩ => rfl
  right_inv _ := rfl


@[simp] lemma rangeInr_apply_inr (α : Type*) {β : Type*} (x : β) :
    (rangeInr α β) ⟨.inr x, mem_range_self _⟩ = x :=
  rfl


/-- If `f : α → β` has a left-inverse when `α` is nonempty, then `α` is computably equivalent to the
range of `f`.

While awkward, the `Nonempty α` hypothesis on `f_inv` and `hf` allows this to be used when `α` is
empty too. This hypothesis is absent on analogous definitions on stronger `Equiv`s like
`LinearEquiv.ofLeftInverse` and `RingEquiv.ofLeftInverse` as their typeclass assumptions
are already sufficient to ensure non-emptiness. -/
@[simps]
def ofLeftInverse {α β : Sort _} (f : α → β) (f_inv : Nonempty α → β → α)
    (hf : ∀ h : Nonempty α, LeftInverse (f_inv h) f) :
    α ≃ range f where
  toFun a := ⟨f a, a, rfl⟩
  invFun b := f_inv (nonempty_of_exists b.2) b
  left_inv a := hf ⟨a⟩ a
  right_inv := fun ⟨b, a, ha⟩ =>
    Subtype.eq <| show f (f_inv ⟨a⟩ b) = b from Eq.trans (congr_arg f <| ha ▸ hf _ a) ha


/-- If `f : α → β` has a left-inverse, then `α` is computably equivalent to the range of `f`.

Note that if `α` is empty, no such `f_inv` exists and so this definition can't be used, unlike
the stronger but less convenient `ofLeftInverse`. -/
abbrev ofLeftInverse' {α β : Sort _} (f : α → β) (f_inv : β → α) (hf : LeftInverse f_inv f) :
    α ≃ range f :=
  ofLeftInverse f (fun _ => f_inv) fun _ => hf


/-- If `f : α → β` is an injective function, then domain `α` is equivalent to the range of `f`. -/
@[simps! apply]
noncomputable def ofInjective {α β} (f : α → β) (hf : Injective f) : α ≃ range f :=
  Equiv.ofLeftInverse f (fun _ => Function.invFun f) fun _ => Function.leftInverse_invFun hf


theorem apply_ofInjective_symm {α β} {f : α → β} (hf : Injective f) (b : range f) :
    f ((ofInjective f hf).symm b) = b :=
  Subtype.ext_iff.1 <| (ofInjective f hf).apply_symm_apply b


@[simp]
theorem ofInjective_symm_apply {α β} {f : α → β} (hf : Injective f) (a : α) :
    (ofInjective f hf).symm ⟨f a, ⟨a, rfl⟩⟩ = a := by
  /-
    α : Sort u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    a : α
    ⊢ Eq ((Equiv.ofInjective f hf).symm ⟨f a, ⋯⟩) a
  -/
  apply (ofInjective f hf).injective
  /-
    case a
    α : Sort u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    a : α
    ⊢ Eq ((Equiv.ofInjective f hf) ((Equiv.ofInjective f hf).symm ⟨f a, ⋯⟩)) ((Equ …
  -/
  simp [apply_ofInjective_symm hf]
  /-
    🎉 no goals
  -/


theorem coe_ofInjective_symm {α β} {f : α → β} (hf : Injective f) :
    ((ofInjective f hf).symm : range f → α) = rangeSplitting f := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    ⊢ Eq (⇑(Equiv.ofInjective f hf).symm) (Set.rangeSplitting f)
  -/
  ext ⟨y, x, rfl⟩
  /-
    case h.mk.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    x : α
    ⊢ Eq ((Equiv.ofInjective f hf).symm ⟨f x, ⋯⟩) (Set.rangeSplitting f ⟨f x, ⋯⟩)
  -/
  apply hf
  /-
    case h.mk.intro.a
    α : Type u_1
    β : Type u_2
    f : α → β
    hf : Function.Injective f
    x : α
    ⊢ Eq (f ((Equiv.ofInjective f hf).symm ⟨f x, ⋯⟩)) (f (Set.rangeSplitting f ⟨f  …
  -/
  simp [apply_rangeSplitting f]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_comp_ofInjective_symm {α β} {f : α → β} (hf : Injective f) :
    f ∘ (ofInjective f hf).symm = Subtype.val :=
  funext fun x => apply_ofInjective_symm hf x


theorem ofLeftInverse_eq_ofInjective {α β : Type*} (f : α → β) (f_inv : Nonempty α → β → α)
    (hf : ∀ h : Nonempty α, LeftInverse (f_inv h) f) :
    ofLeftInverse f f_inv hf =
      ofInjective f ((isEmpty_or_nonempty α).elim (fun _ _ _ _ => Subsingleton.elim _ _)
        (fun h => (hf h).injective)) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    f_inv : Nonempty α → β → α
    hf : ∀ (h : Nonempty α), Function.LeftInverse (f_inv h) f
    ⊢ Eq (Equiv.ofLeftInverse f f_inv hf) (Equiv.ofInjective f ⋯)
  -/
  ext
  /-
    case H.a
    α : Type u_1
    β : Type u_2
    f : α → β
    f_inv : Nonempty α → β → α
    hf : ∀ (h : Nonempty α), Function.LeftInverse (f_inv h) f
    x✝ : α
    ⊢ Eq ↑((Equiv.ofLeftInverse f f_inv hf) x✝) ↑((Equiv.ofInjective f ⋯) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem ofLeftInverse'_eq_ofInjective {α β : Type*} (f : α → β) (f_inv : β → α)
    (hf : LeftInverse f_inv f) : ofLeftInverse' f f_inv hf = ofInjective f hf.injective := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    f_inv : β → α
    hf : Function.LeftInverse f_inv f
    ⊢ Eq (Equiv.ofLeftInverse' f f_inv hf) (Equiv.ofInjective f ⋯)
  -/
  ext
  /-
    case H.a
    α : Type u_1
    β : Type u_2
    f : α → β
    f_inv : β → α
    hf : Function.LeftInverse f_inv f
    x✝ : α
    ⊢ Eq ↑((Equiv.ofLeftInverse' f f_inv hf) x✝) ↑((Equiv.ofInjective f ⋯) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


protected theorem set_forall_iff {α β} (e : α ≃ β) {p : Set α → Prop} :
    (∀ a, p a) ↔ ∀ a, p (e ⁻¹' a) :=
  e.injective.preimage_surjective.forall


theorem preimage_piEquivPiSubtypeProd_symm_pi {α : Type*} {β : α → Type*} (p : α → Prop)
    [DecidablePred p] (s : ∀ i, Set (β i)) :
    (piEquivPiSubtypeProd p β).symm ⁻¹' pi univ s =
      (pi univ fun i : { i // p i } => s i) ×ˢ pi univ fun i : { i // ¬p i } => s i := by
  /-
    α : Type u_1
    β : α → Type u_2
    p : α → Prop
    inst✝ : DecidablePred p
    s : (i : α) → Set (β i)
    ⊢ Eq (Set.preimage (⇑(Equiv.piEquivPiSubtypeProd p β).symm) (_root_.Set.univ.p …
  -/
  ext ⟨f, g⟩
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    p : α → Prop
    inst✝ : DecidablePred p
    s : (i : α) → Set (β i)
    f : (i : Subtype fun x => p x) → β ↑i
    g : (i : Subtype fun x => Not (p x)) → β ↑i
    ⊢ Iff (Membership.mem (Set.preimage (⇑(Equiv.piEquivPiSubtypeProd p β).symm) ( …
  -/
  simp only [mem_preimage, mem_univ_pi, prod_mk_mem_set_prod_eq, Subtype.forall, ← forall_and]
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    p : α → Prop
    inst✝ : DecidablePred p
    s : (i : α) → Set (β i)
    f : (i : Subtype fun x => p x) → β ↑i
    g : (i : Subtype fun x => Not (p x)) → β ↑i
    ⊢ Iff (∀ (i : α), Membership.mem (s i) ((Equiv.piEquivPiSubtypeProd p β).symm  …
  -/
  refine forall_congr' fun i => ?_
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    p : α → Prop
    inst✝ : DecidablePred p
    s : (i : α) → Set (β i)
    f : (i : Subtype fun x => p x) → β ↑i
    g : (i : Subtype fun x => Not (p x)) → β ↑i
    i : α
    ⊢ Iff (Membership.mem (s i) ((Equiv.piEquivPiSubtypeProd p β).symm { fst := f, …
  -/
  dsimp only [Subtype.coe_mk]
  /-
    case h.mk
    α : Type u_1
    β : α → Type u_2
    p : α → Prop
    inst✝ : DecidablePred p
    s : (i : α) → Set (β i)
    f : (i : Subtype fun x => p x) → β ↑i
    g : (i : Subtype fun x => Not (p x)) → β ↑i
    i : α
    ⊢ Iff (Membership.mem (s i) ((Equiv.piEquivPiSubtypeProd p β).symm { fst := f, …
  -/
                        /-
                          🎉 no goals
                        -/
  by_cases hi : p i <;> simp [hi]
                        /-
                          🎉 no goals
                        -/

-- See also `Equiv.sigmaFiberEquiv`.

/-- `sigmaPreimageEquiv f` for `f : α → β` is the natural equivalence between
the type of all preimages of points under `f` and the total space `α`. -/
@[simps!]
def sigmaPreimageEquiv {α β} (f : α → β) : (Σb, f ⁻¹' {b}) ≃ α :=
  sigmaFiberEquiv f

-- See also `Equiv.ofFiberEquiv`.

/-- A family of equivalences between preimages of points gives an equivalence between domains. -/
@[simps!]
def ofPreimageEquiv {α β γ} {f : α → γ} {g : β → γ} (e : ∀ c, f ⁻¹' {c} ≃ g ⁻¹' {c}) : α ≃ β :=
  Equiv.ofFiberEquiv e


theorem ofPreimageEquiv_map {α β γ} {f : α → γ} {g : β → γ} (e : ∀ c, f ⁻¹' {c} ≃ g ⁻¹' {c})
    (a : α) : g (ofPreimageEquiv e a) = f a :=
  Equiv.ofFiberEquiv_map e a


/-- If a function is a bijection between two sets `s` and `t`, then it induces an
equivalence between the types `↥s` and `↥t`. -/
noncomputable def Set.BijOn.equiv {α : Type*} {β : Type*} {s : Set α} {t : Set β} (f : α → β)
    (h : BijOn f s t) : s ≃ t :=
  Equiv.ofBijective _ h.bijective


/-- The composition of an updated function with an equiv on a subtype can be expressed as an
updated function. -/
-- Porting note: replace `s : Set α` and `: s` with `p : α → Prop` and `: Subtype p`, since the
-- former now unfolds syntactically to a less general case of the latter.
theorem dite_comp_equiv_update {α : Type*} {β : Sort*} {γ : Sort*} {p : α → Prop}
    (e : β ≃ Subtype p)
    (v : β → γ) (w : α → γ) (j : β) (x : γ) [DecidableEq β] [DecidableEq α]
    [∀ j, Decidable (p j)] :
    (fun i : α => if h : p i then (Function.update v j x) (e.symm ⟨i, h⟩) else w i) =
      Function.update (fun i : α => if h : p i then v (e.symm ⟨i, h⟩) else w i) (e j) x := by
  /-
    α : Type u_1
    β : Sort u_2
    γ : Sort u_3
    p : α → Prop
    e : Equiv β (Subtype p)
    v : β → γ
    w : α → γ
    j : β
    x : γ
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq α
    inst✝ : (j : α) → Decidable (p j)
    ⊢ Eq (fun i => dite (p i) (fun h => Function.update v j x (e.symm ⟨i, h⟩)) fun …
  -/
  ext i
  /-
    case h
    α : Type u_1
    β : Sort u_2
    γ : Sort u_3
    p : α → Prop
    e : Equiv β (Subtype p)
    v : β → γ
    w : α → γ
    j : β
    x : γ
    inst✝² : DecidableEq β
    inst✝¹ : DecidableEq α
    inst✝ : (j : α) → Decidable (p j)
    i : α
    ⊢ Eq (dite (p i) (fun h => Function.update v j x (e.symm ⟨i, h⟩)) fun h => w i …
  -/
  by_cases h : p i
  · rw [dif_pos h, Function.update_apply_equiv_apply, Equiv.symm_symm,
      Function.update_apply, Function.update_apply, dif_pos h]
    have h_coe : (⟨i, h⟩ : Subtype p) = e j ↔ i = e j :=
      Subtype.ext_iff.trans (by rw [Subtype.coe_mk])
    /-
      case pos
      α : Type u_1
      β : Sort u_2
      γ : Sort u_3
      p : α → Prop
      e : Equiv β (Subtype p)
      v : β → γ
      w : α → γ
      j : β
      x : γ
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq α
      inst✝ : (j : α) → Decidable (p j)
      i : α
      h : p i
      h_coe : Iff (Eq ⟨i, h⟩ (e j)) (Eq i ↑(e j))
      ⊢ Eq (ite (Eq ⟨i, h⟩ (e j)) x (Function.comp v ⇑e.symm ⟨i, h⟩)) (ite (Eq i ↑(e …
    -/
    simp [h_coe]
    /-
      🎉 no goals
    -/
  · have : i ≠ e j := by
      contrapose! h
      have : p (e j : α) := (e j).2
      rwa [← h] at this
    /-
      case neg
      α : Type u_1
      β : Sort u_2
      γ : Sort u_3
      p : α → Prop
      e : Equiv β (Subtype p)
      v : β → γ
      w : α → γ
      j : β
      x : γ
      inst✝² : DecidableEq β
      inst✝¹ : DecidableEq α
      inst✝ : (j : α) → Decidable (p j)
      i : α
      h : Not (p i)
      this : Ne i ↑(e j)
      ⊢ Eq (dite (p i) (fun h => Function.update v j x (e.symm ⟨i, h⟩)) fun h => w i …
    -/
    simp [h, this]
    /-
      🎉 no goals
    -/


theorem Equiv.swap_bijOn_self (hs : a ∈ s ↔ b ∈ s) : BijOn (Equiv.swap a b) s s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    hs : Iff (Membership.mem s a) (Membership.mem s b)
    ⊢ Set.BijOn (⇑(Equiv.swap a b)) s s
  -/
  refine ⟨fun x hx ↦ ?_, (Equiv.injective _).injOn, fun x hx ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Set α
      hs : Iff (Membership.mem s a) (Membership.mem s b)
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem s ((Equiv.swap a b) x)
    -/
  · obtain (rfl | hxa) := eq_or_ne x a
      /-
        case refine_1.inl
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        s : Set α
        x : α
        hx : Membership.mem s x
        hs : Iff (Membership.mem s x) (Membership.mem s b)
        ⊢ Membership.mem s ((Equiv.swap x b) x)
      -/
    · rwa [swap_apply_left, ← hs]
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Set α
      hs : Iff (Membership.mem s a) (Membership.mem s b)
      x : α
      hx : Membership.mem s x
      hxa : Ne x a
      ⊢ Membership.mem s ((Equiv.swap a b) x)
    -/
    obtain (rfl | hxb) := eq_or_ne x b
      /-
        case refine_1.inr.inl
        α : Type u_1
        inst✝ : DecidableEq α
        a : α
        s : Set α
        x : α
        hx : Membership.mem s x
        hxa : Ne x a
        hs : Iff (Membership.mem s a) (Membership.mem s x)
        ⊢ Membership.mem s ((Equiv.swap a x) x)
      -/
    · rwa [swap_apply_right, hs]
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr.inr
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Set α
      hs : Iff (Membership.mem s a) (Membership.mem s b)
      x : α
      hx : Membership.mem s x
      hxa : Ne x a
      hxb : Ne x b
      ⊢ Membership.mem s ((Equiv.swap a b) x)
    -/
    rwa [swap_apply_of_ne_of_ne hxa hxb]
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    hs : Iff (Membership.mem s a) (Membership.mem s b)
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a b)) s) x
  -/
  obtain (rfl | hxa) := eq_or_ne x a
    /-
      case refine_2.inl
      α : Type u_1
      inst✝ : DecidableEq α
      b : α
      s : Set α
      x : α
      hx : Membership.mem s x
      hs : Iff (Membership.mem s x) (Membership.mem s b)
      ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap x b)) s) x
    -/
  · simp [hs.1 hx]
    /-
      🎉 no goals
    -/
  /-
    case refine_2.inr
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    hs : Iff (Membership.mem s a) (Membership.mem s b)
    x : α
    hx : Membership.mem s x
    hxa : Ne x a
    ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a b)) s) x
  -/
  obtain (rfl | hxb) := eq_or_ne x b
    /-
      case refine_2.inr.inl
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Set α
      x : α
      hx : Membership.mem s x
      hxa : Ne x a
      hs : Iff (Membership.mem s a) (Membership.mem s x)
      ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a x)) s) x
    -/
  · simp [hs.2 hx]
    /-
      🎉 no goals
    -/
  /-
    case refine_2.inr.inr
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    hs : Iff (Membership.mem s a) (Membership.mem s b)
    x : α
    hx : Membership.mem s x
    hxa : Ne x a
    hxb : Ne x b
    ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a b)) s) x
  -/
  exact ⟨x, hx, swap_apply_of_ne_of_ne hxa hxb⟩
  /-
    🎉 no goals
  -/


theorem Equiv.swap_bijOn_exchange (ha : a ∈ s) (hb : b ∉ s) :
    BijOn (Equiv.swap a b) s (insert b (s \ {a})) := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    ha : Membership.mem s a
    hb : Not (Membership.mem s b)
    ⊢ Set.BijOn (⇑(Equiv.swap a b)) s (Insert.insert b (SDiff.sdiff s (Singleton.s …
  -/
  refine ⟨fun x hx ↦ ?_, (Equiv.injective _).injOn, fun x hx ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Set α
      ha : Membership.mem s a
      hb : Not (Membership.mem s b)
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem (Insert.insert b (SDiff.sdiff s (Singleton.singleton a))) ((E …
    -/
  · obtain (rfl | hxa) := eq_or_ne x a
      /-
        case refine_1.inl
        α : Type u_1
        inst✝ : DecidableEq α
        b : α
        s : Set α
        hb : Not (Membership.mem s b)
        x : α
        hx ha : Membership.mem s x
        ⊢ Membership.mem (Insert.insert b (SDiff.sdiff s (Singleton.singleton x))) ((E …
      -/
    · simp [swap_apply_left]
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Set α
      ha : Membership.mem s a
      hb : Not (Membership.mem s b)
      x : α
      hx : Membership.mem s x
      hxa : Ne x a
      ⊢ Membership.mem (Insert.insert b (SDiff.sdiff s (Singleton.singleton a))) ((E …
    -/
    rw [swap_apply_of_ne_of_ne hxa (by rintro rfl; contradiction)]
    /-
      case refine_1.inr
      α : Type u_1
      inst✝ : DecidableEq α
      a b : α
      s : Set α
      ha : Membership.mem s a
      hb : Not (Membership.mem s b)
      x : α
      hx : Membership.mem s x
      hxa : Ne x a
      ⊢ Membership.mem (Insert.insert b (SDiff.sdiff s (Singleton.singleton a))) x
    -/
    exact .inr ⟨hx, hxa⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    ha : Membership.mem s a
    hb : Not (Membership.mem s b)
    x : α
    hx : Membership.mem (Insert.insert b (SDiff.sdiff s (Singleton.singleton a))) x
    ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a b)) s) x
  -/
  obtain (rfl | hxb) := eq_or_ne x b
    /-
      case refine_2.inl
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Set α
      ha : Membership.mem s a
      x : α
      hb : Not (Membership.mem s x)
      hx : Membership.mem (Insert.insert x (SDiff.sdiff s (Singleton.singleton a))) x
      ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a x)) s) x
    -/
  · exact ⟨a, ha, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2.inr
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    ha : Membership.mem s a
    hb : Not (Membership.mem s b)
    x : α
    hx : Membership.mem (Insert.insert b (SDiff.sdiff s (Singleton.singleton a))) x
    hxb : Ne x b
    ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a b)) s) x
  -/
  simp only [mem_insert_iff, mem_diff, mem_singleton_iff, or_iff_right hxb] at hx
  /-
    case refine_2.inr
    α : Type u_1
    inst✝ : DecidableEq α
    a b : α
    s : Set α
    ha : Membership.mem s a
    hb : Not (Membership.mem s b)
    x : α
    hxb : Ne x b
    hx : And (Membership.mem s x) (Not (Eq x a))
    ⊢ Membership.mem (_root_.Set.image (⇑(Equiv.swap a b)) s) x
  -/
  exact ⟨x, hx.1, swap_apply_of_ne_of_ne hx.2 hxb⟩
  /-
    🎉 no goals
  -/


