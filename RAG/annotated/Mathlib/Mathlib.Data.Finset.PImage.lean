/-- Convert an `o : Part α` with decidable `Part.Dom o` to `Finset α`. -/
def toFinset (o : Part α) [Decidable o.Dom] : Finset α :=
  o.toOption.toFinset


@[simp]
theorem mem_toFinset {o : Part α} [Decidable o.Dom] {x : α} : x ∈ o.toFinset ↔ x ∈ o := by
  /-
    α : Type u_1
    o : Part α
    inst✝ : Decidable o.Dom
    x : α
    ⊢ Iff (Membership.mem o.toFinset x) (Membership.mem o x)
  -/
  simp [toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_none [Decidable (none : Part α).Dom] : none.toFinset = (∅ : Finset α) := by
  /-
    α : Type u_1
    inst✝ : Decidable Part.none.Dom
    ⊢ Eq Part.none.toFinset EmptyCollection.emptyCollection
  -/
  simp [toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem toFinset_some {a : α} [Decidable (some a).Dom] : (some a).toFinset = {a} := by
  /-
    α : Type u_1
    a : α
    inst✝ : Decidable (Part.some a).Dom
    ⊢ Eq (Part.some a).toFinset (Singleton.singleton a)
  -/
  simp [toFinset]
  /-
    🎉 no goals
  -/


@[simp]
theorem coe_toFinset (o : Part α) [Decidable o.Dom] : (o.toFinset : Set α) = { x | x ∈ o } :=
  Set.ext fun _ => mem_toFinset


/-- Image of `s : Finset α` under a partially defined function `f : α →. β`. -/
def pimage (f : α →. β) [∀ x, Decidable (f x).Dom] (s : Finset α) : Finset β :=
  s.biUnion fun x => (f x).toFinset


@[simp]
theorem mem_pimage : b ∈ s.pimage f ↔ ∃ a ∈ s, b ∈ f a := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    s : Finset α
    b : β
    ⊢ Iff (Membership.mem (Finset.pimage f s) b) (Exists fun a => And (Membership. …
  -/
  simp [pimage]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_pimage : (s.pimage f : Set β) = f.image s :=
  Set.ext fun _ => mem_pimage


@[simp]
theorem pimage_some (s : Finset α) (f : α → β) [∀ x, Decidable (Part.some <| f x).Dom] :
    (s.pimage fun x => Part.some (f x)) = s.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    s : Finset α
    f : α → β
    inst✝ : (x : α) → Decidable (Part.some (f x)).Dom
    ⊢ Eq (Finset.pimage (fun x => Part.some (f x)) s) (Finset.image f s)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    s : Finset α
    f : α → β
    inst✝ : (x : α) → Decidable (Part.some (f x)).Dom
    a✝ : β
    ⊢ Iff (Membership.mem (Finset.pimage (fun x => Part.some (f x)) s) a✝) (Member …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


theorem pimage_congr (h₁ : s = t) (h₂ : ∀ x ∈ t, f x = g x) : s.pimage f = t.pimage g := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq β
    f g : PFun α β
    inst✝¹ : (x : α) → Decidable (f x).Dom
    inst✝ : (x : α) → Decidable (g x).Dom
    s t : Finset α
    h₁ : Eq s t
    h₂ : ∀ (x : α), Membership.mem t x → Eq (f x) (g x)
    ⊢ Eq (Finset.pimage f s) (Finset.pimage g t)
  -/
  subst s
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq β
    f g : PFun α β
    inst✝¹ : (x : α) → Decidable (f x).Dom
    inst✝ : (x : α) → Decidable (g x).Dom
    t : Finset α
    h₂ : ∀ (x : α), Membership.mem t x → Eq (f x) (g x)
    ⊢ Eq (Finset.pimage f t) (Finset.pimage g t)
  -/
  ext y
  -- Porting note: `← exists_prop` required because `∃ x ∈ s, p x` is defined differently
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq β
    f g : PFun α β
    inst✝¹ : (x : α) → Decidable (f x).Dom
    inst✝ : (x : α) → Decidable (g x).Dom
    t : Finset α
    h₂ : ∀ (x : α), Membership.mem t x → Eq (f x) (g x)
    y : β
    ⊢ Iff (Membership.mem (Finset.pimage f t) y) (Membership.mem (Finset.pimage g  …
  -/
  simp +contextual only [mem_pimage, ← exists_prop, h₂]
  /-
    🎉 no goals
  -/


/-- Rewrite `s.pimage f` in terms of `Finset.filter`, `Finset.attach`, and `Finset.image`. -/
theorem pimage_eq_image_filter : s.pimage f =
    (filter (fun x => (f x).Dom) s).attach.image
      fun x : { x // x ∈ filter (fun x => (f x).Dom) s } =>
        (f x).get (mem_filter.mp x.coe_prop).2 := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    s : Finset α
    ⊢ Eq (Finset.pimage f s) (Finset.image (fun x => (f ↑x).get ⋯) (Finset.filter  …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    s : Finset α
    x : β
    ⊢ Iff (Membership.mem (Finset.pimage f s) x) (Membership.mem (Finset.image (fu …
  -/
  simp [Part.mem_eq, And.exists]
  -- Porting note: `← exists_prop` required because `∃ x ∈ s, p x` is defined differently
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    s : Finset α
    x : β
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Exists fun h => Eq ((f a).get …
  -/
  simp only [← exists_prop]
  /-
    🎉 no goals
  -/


theorem pimage_union [DecidableEq α] : (s ∪ t).pimage f = s.pimage f ∪ t.pimage f :=
  coe_inj.1 <| by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq β
    f : PFun α β
    inst✝¹ : (x : α) → Decidable (f x).Dom
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ Eq ↑(Finset.pimage f (Union.union s t)) ↑(Union.union (Finset.pimage f s) (F …
  -/
  simp only [coe_pimage, coe_union, ← PFun.image_union]
  /-
    🎉 no goals
  -/


@[simp]
theorem pimage_empty : pimage f ∅ = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    ⊢ Eq (Finset.pimage f EmptyCollection.emptyCollection) EmptyCollection.emptyCo …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    a✝ : β
    ⊢ Iff (Membership.mem (Finset.pimage f EmptyCollection.emptyCollection) a✝) (M …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem pimage_subset {t : Finset β} : s.pimage f ⊆ t ↔ ∀ x ∈ s, ∀ y ∈ f x, y ∈ t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    f : PFun α β
    inst✝ : (x : α) → Decidable (f x).Dom
    s : Finset α
    t : Finset β
    ⊢ Iff (HasSubset.Subset (Finset.pimage f s) t) (∀ (x : α), Membership.mem s x  …
  -/
  simp [subset_iff, @forall_swap _ β]
  /-
    🎉 no goals
  -/


@[mono]
theorem pimage_mono (h : s ⊆ t) : s.pimage f ⊆ t.pimage f :=
  pimage_subset.2 fun x hx _ hy => mem_pimage.2 ⟨x, h hx, hy⟩


theorem pimage_inter [DecidableEq α] : (s ∩ t).pimage f ⊆ s.pimage f ∩ t.pimage f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝² : DecidableEq β
    f : PFun α β
    inst✝¹ : (x : α) → Decidable (f x).Dom
    s t : Finset α
    inst✝ : DecidableEq α
    ⊢ HasSubset.Subset (Finset.pimage f (Inter.inter s t)) (Inter.inter (Finset.pi …
  -/
  simp only [← coe_subset, coe_pimage, coe_inter, PFun.image_inter]
  /-
    🎉 no goals
  -/


