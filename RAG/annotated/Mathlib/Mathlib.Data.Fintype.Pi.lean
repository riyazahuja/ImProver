/-- Given for all `a : α` a finset `t a` of `δ a`, then one can define the
finset `Fintype.piFinset t` of all functions taking values in `t a` for all `a`. This is the
analogue of `Finset.pi` where the base finset is `univ` (but formally they are not the same, as
there is an additional condition `i ∈ Finset.univ` in the `Finset.pi` definition). -/
def piFinset (t : ∀ a, Finset (δ a)) : Finset (∀ a, δ a) :=
  (Finset.univ.pi t).map ⟨fun f a => f a (mem_univ a), fun _ _ =>
       /-
         α : Type u_1
         β : Type u_2
         inst✝¹ : DecidableEq α
         inst✝ : Fintype α
         γ : α → Type u_3
         δ : α → Type u_4
         s : (a : α) → Finset (γ a)
         t : (a : α) → Finset (δ a)
         x✝¹ x✝ : (a : α) → Membership.mem Finset.univ a → δ a
         ⊢ Eq ((fun f a => f a ⋯) x✝¹) ((fun f a => f a ⋯) x✝) → Eq x✝¹ x✝
       -/
    by simp (config := {contextual := true}) [funext_iff]⟩
       /-
         🎉 no goals
       -/


@[simp]
theorem mem_piFinset {t : ∀ a, Finset (δ a)} {f : ∀ a, δ a} : f ∈ piFinset t ↔ ∀ a, f a ∈ t a := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    δ : α → Type u_4
    t : (a : α) → Finset (δ a)
    f : (a : α) → δ a
    ⊢ Iff (Membership.mem (Fintype.piFinset t) f) (∀ (a : α), Membership.mem (t a) …
  -/
  constructor
  · simp only [piFinset, mem_map, and_imp, forall_prop_of_true, exists_prop, mem_univ, exists_imp,
      mem_pi]
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      f : (a : α) → δ a
      ⊢ ∀ (x : (a : α) → Membership.mem Finset.univ a → δ a), (∀ (a : α), Membership …
    -/
    rintro g hg hgf a
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      f : (a : α) → δ a
      g : (a : α) → Membership.mem Finset.univ a → δ a
      hg : ∀ (a : α), Membership.mem (t a) (g a ⋯)
      hgf : Eq ({ toFun := fun f a => f a ⋯, inj' := ⋯ } g) f
      a : α
      ⊢ Membership.mem (t a) (f a)
    -/
    rw [← hgf]
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      f : (a : α) → δ a
      g : (a : α) → Membership.mem Finset.univ a → δ a
      hg : ∀ (a : α), Membership.mem (t a) (g a ⋯)
      hgf : Eq ({ toFun := fun f a => f a ⋯, inj' := ⋯ } g) f
      a : α
      ⊢ Membership.mem (t a) ({ toFun := fun f a => f a ⋯, inj' := ⋯ } g a)
    -/
    exact hg a
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      f : (a : α) → δ a
      ⊢ (∀ (a : α), Membership.mem (t a) (f a)) → Membership.mem (Fintype.piFinset t …
    -/
  · simp only [piFinset, mem_map, forall_prop_of_true, exists_prop, mem_univ, mem_pi]
    /-
      case mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      f : (a : α) → δ a
      ⊢ (∀ (a : α), Membership.mem (t a) (f a)) → Exists fun a => And (∀ (a_1 : α),  …
    -/
    exact fun hf => ⟨fun a _ => f a, hf, rfl⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_piFinset (t : ∀ a, Finset (δ a)) :
    (piFinset t : Set (∀ a, δ a)) = Set.pi Set.univ fun a => t a :=
  Set.ext fun x => by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      x : (a : α) → δ a
      ⊢ Iff (Membership.mem (↑(Fintype.piFinset t)) x) (Membership.mem (Set.univ.pi  …
    -/
    rw [Set.mem_univ_pi]
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      δ : α → Type u_4
      t : (a : α) → Finset (δ a)
      x : (a : α) → δ a
      ⊢ Iff (Membership.mem (↑(Fintype.piFinset t)) x) (∀ (i : α), Membership.mem (↑ …
    -/
    exact Fintype.mem_piFinset
    /-
      🎉 no goals
    -/


theorem piFinset_subset (t₁ t₂ : ∀ a, Finset (δ a)) (h : ∀ a, t₁ a ⊆ t₂ a) :
    piFinset t₁ ⊆ piFinset t₂ := fun _ hg => mem_piFinset.2 fun a => h a <| mem_piFinset.1 hg a


@[simp]
                                                                /-
                                                                  α : Type u_1
                                                                  inst✝¹ : DecidableEq α
                                                                  inst✝ : Fintype α
                                                                  γ : α → Type u_3
                                                                  s : (a : α) → Finset (γ a)
                                                                  ⊢ Iff (Eq (Fintype.piFinset s) EmptyCollection.emptyCollection) (Exists fun i  …
                                                                -/
theorem piFinset_eq_empty : piFinset s = ∅ ↔ ∃ i, s i = ∅ := by simp [piFinset]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
                                                                                          /-
                                                                                            α : Type u_1
                                                                                            inst✝² : DecidableEq α
                                                                                            inst✝¹ : Fintype α
                                                                                            δ : α → Type u_4
                                                                                            inst✝ : Nonempty α
                                                                                            ⊢ Eq (Fintype.piFinset fun x => EmptyCollection.emptyCollection) EmptyCollecti …
                                                                                          -/
theorem piFinset_empty [Nonempty α] : piFinset (fun _ => ∅ : ∀ i, Finset (δ i)) = ∅ := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


@[simp]
                                                                            /-
                                                                              α : Type u_1
                                                                              inst✝¹ : DecidableEq α
                                                                              inst✝ : Fintype α
                                                                              γ : α → Type u_3
                                                                              s : (a : α) → Finset (γ a)
                                                                              ⊢ Iff (Fintype.piFinset s).Nonempty (∀ (a : α), (s a).Nonempty)
                                                                            -/
lemma piFinset_nonempty : (piFinset s).Nonempty ↔ ∀ a, (s a).Nonempty := by simp [piFinset]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
alias ⟨_, Aesop.piFinset_nonempty_of_forall_nonempty⟩ := piFinset_nonempty


lemma _root_.Finset.Nonempty.piFinset_const {ι : Type*} [Fintype ι] [DecidableEq ι] {s : Finset β}
    (hs : s.Nonempty) : (piFinset fun _ : ι ↦ s).Nonempty := piFinset_nonempty.2 fun _ ↦ hs


@[simp]
lemma piFinset_of_isEmpty [IsEmpty α] (s : ∀ a, Finset (γ a)) : piFinset s = univ :=
                               /-
                                 α : Type u_1
                                 inst✝² : DecidableEq α
                                 inst✝¹ : Fintype α
                                 γ : α → Type u_3
                                 inst✝ : IsEmpty α
                                 s : (a : α) → Finset (γ a)
                                 x✝ : (a : α) → γ a
                                 ⊢ Membership.mem (Fintype.piFinset s) x✝
                               -/
  eq_univ_of_forall fun _ ↦ by simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem piFinset_singleton (f : ∀ i, δ i) : piFinset (fun i => {f i} : ∀ i, Finset (δ i)) = {f} :=
                  /-
                    α : Type u_1
                    inst✝¹ : DecidableEq α
                    inst✝ : Fintype α
                    δ : α → Type u_4
                    f x✝ : (a : α) → δ a
                    ⊢ Iff (Membership.mem (Fintype.piFinset fun i => Singleton.singleton (f i)) x✝ …
                  -/
  ext fun _ => by simp only [funext_iff, Fintype.mem_piFinset, mem_singleton]
                  /-
                    🎉 no goals
                  -/


theorem piFinset_subsingleton {f : ∀ i, Finset (δ i)} (hf : ∀ i, (f i : Set (δ i)).Subsingleton) :
    (Fintype.piFinset f : Set (∀ i, δ i)).Subsingleton := fun _ ha _ hb =>
  funext fun _ => hf _ (mem_piFinset.1 ha _) (mem_piFinset.1 hb _)


theorem piFinset_disjoint_of_disjoint (t₁ t₂ : ∀ a, Finset (δ a)) {a : α}
    (h : Disjoint (t₁ a) (t₂ a)) : Disjoint (piFinset t₁) (piFinset t₂) :=
  disjoint_iff_ne.2 fun f₁ hf₁ f₂ hf₂ eq₁₂ =>
    disjoint_iff_ne.1 h (f₁ a) (mem_piFinset.1 hf₁ a) (f₂ a) (mem_piFinset.1 hf₂ a)
      (congr_fun eq₁₂ a)


lemma piFinset_image [∀ a, DecidableEq (δ a)] (f : ∀ a, γ a → δ a) (s : ∀ a, Finset (γ a)) :
    piFinset (fun a ↦ (s a).image (f a)) = (piFinset s).image fun b a ↦ f _ (b a) := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    γ : α → Type u_3
    δ : α → Type u_4
    inst✝ : (a : α) → DecidableEq (δ a)
    f : (a : α) → γ a → δ a
    s : (a : α) → Finset (γ a)
    ⊢ Eq (Fintype.piFinset fun a => Finset.image (f a) (s a)) (Finset.image (fun b …
  -/
  ext; simp only [mem_piFinset, mem_image, Classical.skolem, forall_and, funext_iff]
       /-
         🎉 no goals
       -/


lemma eval_image_piFinset_subset (t : ∀ a, Finset (δ a)) (a : α) [DecidableEq (δ a)] :
    ((piFinset t).image fun f ↦ f a) ⊆ t a := image_subset_iff.2 fun _x hx ↦ mem_piFinset.1 hx _


lemma eval_image_piFinset (t : ∀ a, Finset (δ a)) (a : α) [DecidableEq (δ a)]
    (ht : ∀ b, a ≠ b → (t b).Nonempty) : ((piFinset t).image fun f ↦ f a) = t a := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    t : (a : α) → Finset (δ a)
    a : α
    inst✝ : DecidableEq (δ a)
    ht : ∀ (b : α), Ne a b → (t b).Nonempty
    ⊢ Eq (Finset.image (fun f => f a) (Fintype.piFinset t)) (t a)
  -/
  refine (eval_image_piFinset_subset _ _).antisymm fun x h ↦ mem_image.2 ?_
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    t : (a : α) → Finset (δ a)
    a : α
    inst✝ : DecidableEq (δ a)
    ht : ∀ (b : α), Ne a b → (t b).Nonempty
    x : δ a
    h : Membership.mem (t a) x
    ⊢ Exists fun a_1 => And (Membership.mem (Fintype.piFinset t) a_1) (Eq (a_1 a) x)
  -/
  choose f hf using ht
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    t : (a : α) → Finset (δ a)
    a : α
    inst✝ : DecidableEq (δ a)
    x : δ a
    h : Membership.mem (t a) x
    f : (b : α) → Ne a b → δ b
    hf : ∀ (b : α) (a : Ne a b), Membership.mem (t b) (f b a)
    ⊢ Exists fun a_1 => And (Membership.mem (Fintype.piFinset t) a_1) (Eq (a_1 a) x)
  -/
  exact ⟨fun b ↦ if h : a = b then h ▸ x else f _ h, by aesop, by simp⟩
  /-
    🎉 no goals
  -/


lemma eval_image_piFinset_const {β} [DecidableEq β] (t : Finset β) (a : α) :
    ((piFinset fun _i : α ↦ t).image fun f ↦ f a) = t := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    β : Type u_5
    inst✝ : DecidableEq β
    t : Finset β
    a : α
    ⊢ Eq (Finset.image (fun f => f a) (Fintype.piFinset fun _i => t)) t
  -/
  obtain rfl | ht := t.eq_empty_or_nonempty
    /-
      case inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      β : Type u_5
      inst✝ : DecidableEq β
      a : α
      ⊢ Eq (Finset.image (fun f => f a) (Fintype.piFinset fun _i => EmptyCollection. …
    -/
  · haveI : Nonempty α := ⟨a⟩
    /-
      case inl
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      β : Type u_5
      inst✝ : DecidableEq β
      a : α
      this : Nonempty α
      ⊢ Eq (Finset.image (fun f => f a) (Fintype.piFinset fun _i => EmptyCollection. …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      β : Type u_5
      inst✝ : DecidableEq β
      t : Finset β
      a : α
      ht : t.Nonempty
      ⊢ Eq (Finset.image (fun f => f a) (Fintype.piFinset fun _i => t)) t
    -/
  · exact eval_image_piFinset (fun _ ↦ t) a fun _ _ ↦ ht
    /-
      🎉 no goals
    -/


lemma filter_piFinset_of_not_mem (t : ∀ a, Finset (δ a)) (a : α) (x : δ a) (hx : x ∉ t a) :
    {f ∈ piFinset t | f a = x} = ∅ := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    inst✝ : (a : α) → DecidableEq (δ a)
    t : (a : α) → Finset (δ a)
    a : α
    x : δ a
    hx : Not (Membership.mem (t a) x)
    ⊢ Eq (Finset.filter (fun f => Eq (f a) x) (Fintype.piFinset t)) EmptyCollectio …
  -/
  simp only [filter_eq_empty_iff, mem_piFinset]; rintro f hf rfl; exact hx (hf _)
                                                                  /-
                                                                    🎉 no goals
                                                                  -/

-- TODO: This proof looks like a good example of something that `aesop` can't do but should

lemma piFinset_update_eq_filter_piFinset_mem (s : ∀ i, Finset (δ i)) (i : α) {t : Finset (δ i)}
    (hts : t ⊆ s i) : piFinset (Function.update s i t) = {f ∈ piFinset s | f i ∈ t} := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    inst✝ : (a : α) → DecidableEq (δ a)
    s : (i : α) → Finset (δ i)
    i : α
    t : Finset (δ i)
    hts : HasSubset.Subset t (s i)
    ⊢ Eq (Fintype.piFinset (Function.update s i t)) (Finset.filter (fun f => Membe …
  -/
  ext f
  /-
    case h
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    inst✝ : (a : α) → DecidableEq (δ a)
    s : (i : α) → Finset (δ i)
    i : α
    t : Finset (δ i)
    hts : HasSubset.Subset t (s i)
    f : (a : α) → δ a
    ⊢ Iff (Membership.mem (Fintype.piFinset (Function.update s i t)) f) (Membershi …
  -/
  simp only [mem_piFinset, mem_filter]
  /-
    case h
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    inst✝ : (a : α) → DecidableEq (δ a)
    s : (i : α) → Finset (δ i)
    i : α
    t : Finset (δ i)
    hts : HasSubset.Subset t (s i)
    f : (a : α) → δ a
    ⊢ Iff (∀ (a : α), Membership.mem (Function.update s i t a) (f a)) (And (∀ (a : …
  -/
  refine ⟨fun h ↦ ?_, fun h j ↦ ?_⟩
    /-
      case h.refine_1
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      δ : α → Type u_4
      inst✝ : (a : α) → DecidableEq (δ a)
      s : (i : α) → Finset (δ i)
      i : α
      t : Finset (δ i)
      hts : HasSubset.Subset t (s i)
      f : (a : α) → δ a
      h : ∀ (a : α), Membership.mem (Function.update s i t a) (f a)
      ⊢ And (∀ (a : α), Membership.mem (s a) (f a)) (Membership.mem t (f i))
    -/
  · have := by simpa using h i
    /-
      case h.refine_1
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      δ : α → Type u_4
      inst✝ : (a : α) → DecidableEq (δ a)
      s : (i : α) → Finset (δ i)
      i : α
      t : Finset (δ i)
      hts : HasSubset.Subset t (s i)
      f : (a : α) → δ a
      h : ∀ (a : α), Membership.mem (Function.update s i t a) (f a)
      this : Membership.mem t (f i)
      ⊢ And (∀ (a : α), Membership.mem (s a) (f a)) (Membership.mem t (f i))
    -/
    refine ⟨fun j ↦ ?_, this⟩
    /-
      case h.refine_1
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      δ : α → Type u_4
      inst✝ : (a : α) → DecidableEq (δ a)
      s : (i : α) → Finset (δ i)
      i : α
      t : Finset (δ i)
      hts : HasSubset.Subset t (s i)
      f : (a : α) → δ a
      h : ∀ (a : α), Membership.mem (Function.update s i t a) (f a)
      this : Membership.mem t (f i)
      j : α
      ⊢ Membership.mem (s j) (f j)
    -/
    obtain rfl | hji := eq_or_ne j i
      /-
        case h.refine_1.inl
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        δ : α → Type u_4
        inst✝ : (a : α) → DecidableEq (δ a)
        s : (i : α) → Finset (δ i)
        f : (a : α) → δ a
        j : α
        t : Finset (δ j)
        hts : HasSubset.Subset t (s j)
        h : ∀ (a : α), Membership.mem (Function.update s j t a) (f a)
        this : Membership.mem t (f j)
        ⊢ Membership.mem (s j) (f j)
      -/
    · exact hts this
      /-
        🎉 no goals
      -/
      /-
        case h.refine_1.inr
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        δ : α → Type u_4
        inst✝ : (a : α) → DecidableEq (δ a)
        s : (i : α) → Finset (δ i)
        i : α
        t : Finset (δ i)
        hts : HasSubset.Subset t (s i)
        f : (a : α) → δ a
        h : ∀ (a : α), Membership.mem (Function.update s i t a) (f a)
        this : Membership.mem t (f i)
        j : α
        hji : Ne j i
        ⊢ Membership.mem (s j) (f j)
      -/
    · simpa [hji] using h j
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      α : Type u_1
      inst✝² : DecidableEq α
      inst✝¹ : Fintype α
      δ : α → Type u_4
      inst✝ : (a : α) → DecidableEq (δ a)
      s : (i : α) → Finset (δ i)
      i : α
      t : Finset (δ i)
      hts : HasSubset.Subset t (s i)
      f : (a : α) → δ a
      h : And (∀ (a : α), Membership.mem (s a) (f a)) (Membership.mem t (f i))
      j : α
      ⊢ Membership.mem (Function.update s i t j) (f j)
    -/
  · obtain rfl | hji := eq_or_ne j i
      /-
        case h.refine_2.inl
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        δ : α → Type u_4
        inst✝ : (a : α) → DecidableEq (δ a)
        s : (i : α) → Finset (δ i)
        f : (a : α) → δ a
        j : α
        t : Finset (δ j)
        hts : HasSubset.Subset t (s j)
        h : And (∀ (a : α), Membership.mem (s a) (f a)) (Membership.mem t (f j))
        ⊢ Membership.mem (Function.update s j t j) (f j)
      -/
    · simpa using h.2
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.inr
        α : Type u_1
        inst✝² : DecidableEq α
        inst✝¹ : Fintype α
        δ : α → Type u_4
        inst✝ : (a : α) → DecidableEq (δ a)
        s : (i : α) → Finset (δ i)
        i : α
        t : Finset (δ i)
        hts : HasSubset.Subset t (s i)
        f : (a : α) → δ a
        h : And (∀ (a : α), Membership.mem (s a) (f a)) (Membership.mem t (f i))
        j : α
        hji : Ne j i
        ⊢ Membership.mem (Function.update s i t j) (f j)
      -/
    · simpa [hji] using h.1 j
      /-
        🎉 no goals
      -/


lemma piFinset_update_singleton_eq_filter_piFinset_eq (s : ∀ i, Finset (δ i)) (i : α) {a : δ i}
    (ha : a ∈ s i) :
    piFinset (Function.update s i {a}) = {f ∈ piFinset s | f i = a} := by
  /-
    α : Type u_1
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    δ : α → Type u_4
    inst✝ : (a : α) → DecidableEq (δ a)
    s : (i : α) → Finset (δ i)
    i : α
    a : δ i
    ha : Membership.mem (s i) a
    ⊢ Eq (Fintype.piFinset (Function.update s i (Singleton.singleton a))) (Finset. …
  -/
  simp [piFinset_update_eq_filter_piFinset_mem, ha]
  /-
    🎉 no goals
  -/


/-- A dependent product of fintypes, indexed by a fintype, is a fintype. -/
instance Pi.instFintype {α : Type*} {β : α → Type*} [DecidableEq α] [Fintype α]
    [∀ a, Fintype (β a)] : Fintype (∀ a, β a) :=
                                      /-
                                        α✝ : Type u_1
                                        β✝ : Type u_2
                                        α : Type u_3
                                        β : α → Type u_4
                                        inst✝² : DecidableEq α
                                        inst✝¹ : Fintype α
                                        inst✝ : (a : α) → Fintype (β a)
                                        ⊢ ∀ (x : (a : α) → β a), Membership.mem (Fintype.piFinset fun x => Finset.univ …
                                      -/
  ⟨Fintype.piFinset fun _ => univ, by simp⟩
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem Fintype.piFinset_univ {α : Type*} {β : α → Type*} [DecidableEq α] [Fintype α]
    [∀ a, Fintype (β a)] :
    (Fintype.piFinset fun a : α => (Finset.univ : Finset (β a))) =
      (Finset.univ : Finset (∀ a, β a)) :=
  rfl

-- Porting note: this instance used to be computable in Lean3 and used `decidable_eq`, but
-- it makes things a lot harder to work with here. in some ways that was because in Lean3
-- we could make this instance irreducible when needed and in the worst case use `congr/convert`,
-- but those don't work with subsingletons in lean4 as-is so we cannot do this here.

noncomputable instance _root_.Function.Embedding.fintype {α β} [Fintype α] [Fintype β] :
  Fintype (α ↪ β) := by
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    α : Type ?u.25492
    β : Type ?u.25495
    inst✝¹ : Fintype α
    inst✝ : Fintype β
    ⊢ Fintype (Function.Embedding α β)
  -/
  classical exact Fintype.ofEquiv _ (Equiv.subtypeInjectiveEquivEmbedding α β)
  /-
    🎉 no goals
  -/


instance RelHom.instFintype {α β} [Fintype α] [Fintype β] [DecidableEq α] {r : α → α → Prop}
    {s : β → β → Prop} [DecidableRel r] [DecidableRel s] : Fintype (r →r s) :=
  Fintype.ofEquiv {f : α → β // ∀ {x y}, r x y → s (f x) (f y)} <| Equiv.mk
    (fun f ↦ ⟨f.1, f.2⟩) (fun f ↦ ⟨f.1, f.2⟩) (fun _ ↦ rfl) (fun _ ↦ rfl)


noncomputable instance RelEmbedding.instFintype {α β} [Fintype α] [Fintype β]
    {r : α → α → Prop} {s : β → β → Prop} : Fintype (r ↪r s) :=
  Fintype.ofInjective _ RelEmbedding.toEmbedding_injective


@[simp]
theorem Finset.univ_pi_univ {α : Type*} {β : α → Type*} [DecidableEq α] [Fintype α]
    [∀ a, Fintype (β a)] :
    (Finset.univ.pi fun a : α => (Finset.univ : Finset (β a))) = Finset.univ := by
  /-
    α : Type u_3
    β : α → Type u_4
    inst✝² : DecidableEq α
    inst✝¹ : Fintype α
    inst✝ : (a : α) → Fintype (β a)
    ⊢ Eq (Finset.univ.pi fun a => Finset.univ) Finset.univ
  -/
  ext; simp
       /-
         🎉 no goals
       -/


lemma piFinset_filter_const [DecidableEq ι] [Fintype ι] :
                                                                                     /-
                                                                                       α : Type u_1
                                                                                       ι : Type u_3
                                                                                       inst✝² : DecidableEq (ι → α)
                                                                                       s : Finset α
                                                                                       inst✝¹ : DecidableEq ι
                                                                                       inst✝ : Fintype ι
                                                                                       ⊢ Eq (Finset.filter (fun f => Exists fun a => And (Membership.mem s a) (Eq (Fu …
                                                                                     -/
    {f ∈ Fintype.piFinset fun _ : ι ↦ s | ∃ a ∈ s, const ι a = f} = s.piDiag ι := by aesop
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


lemma piDiag_subset_piFinset [DecidableEq ι] [Fintype ι] :
                                                  /-
                                                    α : Type u_1
                                                    ι : Type u_3
                                                    inst✝² : DecidableEq (ι → α)
                                                    s : Finset α
                                                    inst✝¹ : DecidableEq ι
                                                    inst✝ : Fintype ι
                                                    ⊢ HasSubset.Subset (s.piDiag ι) (Fintype.piFinset fun x => s)
                                                  -/
    s.piDiag ι ⊆ Fintype.piFinset fun _ ↦ s := by simp [← piFinset_filter_const]
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- Finite product of finite sets is finite -/
theorem Finite.pi (ht : ∀ i, (t i).Finite) : (pi univ t).Finite := by
  /-
    ι : Type u_3
    inst✝ : Finite ι
    κ : ι → Type u_4
    t : (i : ι) → Set (κ i)
    ht : ∀ (i : ι), (t i).Finite
    ⊢ (Set.univ.pi t).Finite
  -/
  cases nonempty_fintype ι
  /-
    case intro
    ι : Type u_3
    inst✝ : Finite ι
    κ : ι → Type u_4
    t : (i : ι) → Set (κ i)
    ht : ∀ (i : ι), (t i).Finite
    val✝ : Fintype ι
    ⊢ (Set.univ.pi t).Finite
  -/
  lift t to ∀ d, Finset (κ d) using ht
  classical
    rw [← Fintype.coe_piFinset]
    apply Finset.finite_toSet


/-- Finite product of finite sets is finite. Note this is a variant of `Set.Finite.pi` without the
extra `i ∈ univ` binder. -/
lemma Finite.pi' (ht : ∀ i, (t i).Finite) : {f : ∀ i, κ i | ∀ i, f i ∈ t i}.Finite := by
  /-
    ι : Type u_3
    inst✝ : Finite ι
    κ : ι → Type u_4
    t : (i : ι) → Set (κ i)
    ht : ∀ (i : ι), (t i).Finite
    ⊢ (setOf fun f => ∀ (i : ι), Membership.mem (t i) (f i)).Finite
  -/
  simpa [Set.pi] using Finite.pi ht
  /-
    🎉 no goals
  -/


theorem forall_finite_image_eval_iff {δ : Type*} [Finite δ] {κ : δ → Type*} {s : Set (∀ d, κ d)} :
    (∀ d, (eval d '' s).Finite) ↔ s.Finite :=
  ⟨fun h => (Finite.pi h).subset <| subset_pi_eval_image _ _, fun h _ => h.image _⟩


