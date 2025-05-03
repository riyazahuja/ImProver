theorem Subsingleton.prod (hs : s.Subsingleton) (ht : t.Subsingleton) :
    (s ×ˢ t).Subsingleton := fun _x hx _y hy ↦
  Prod.ext (hs hx.1 hy.1) (ht hx.2 hy.2)


noncomputable instance decidableMemProd [DecidablePred (· ∈ s)] [DecidablePred (· ∈ t)] :
    DecidablePred (· ∈ s ×ˢ t) := fun x => inferInstanceAs (Decidable (x.1 ∈ s ∧ x.2 ∈ t))


@[gcongr]
theorem prod_mono (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) : s₁ ×ˢ t₁ ⊆ s₂ ×ˢ t₂ :=
  fun _ ⟨h₁, h₂⟩ => ⟨hs h₁, ht h₂⟩


@[gcongr]
theorem prod_mono_left (hs : s₁ ⊆ s₂) : s₁ ×ˢ t ⊆ s₂ ×ˢ t :=
  prod_mono hs Subset.rfl


@[gcongr]
theorem prod_mono_right (ht : t₁ ⊆ t₂) : s ×ˢ t₁ ⊆ s ×ˢ t₂ :=
  prod_mono Subset.rfl ht


@[simp]
theorem prod_self_subset_prod_self : s₁ ×ˢ s₁ ⊆ s₂ ×ˢ s₂ ↔ s₁ ⊆ s₂ :=
  ⟨fun h _ hx => (h (mk_mem_prod hx hx)).1, fun h _ hx => ⟨h hx.1, h hx.2⟩⟩


@[simp]
theorem prod_self_ssubset_prod_self : s₁ ×ˢ s₁ ⊂ s₂ ×ˢ s₂ ↔ s₁ ⊂ s₂ :=
  and_congr prod_self_subset_prod_self <| not_congr prod_self_subset_prod_self


theorem prod_subset_iff {P : Set (α × β)} : s ×ˢ t ⊆ P ↔ ∀ x ∈ s, ∀ y ∈ t, (x, y) ∈ P :=
  ⟨fun h _ hx _ hy => h (mk_mem_prod hx hy), fun h ⟨_, _⟩ hp => h _ hp.1 _ hp.2⟩


theorem forall_prod_set {p : α × β → Prop} : (∀ x ∈ s ×ˢ t, p x) ↔ ∀ x ∈ s, ∀ y ∈ t, p (x, y) :=
  prod_subset_iff


theorem exists_prod_set {p : α × β → Prop} : (∃ x ∈ s ×ˢ t, p x) ↔ ∃ x ∈ s, ∃ y ∈ t, p (x, y) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    p : Prod α β → Prop
    ⊢ Iff (Exists fun x => And (Membership.mem (SProd.sprod s t) x) (p x)) (Exists …
  -/
  simp [and_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_empty : s ×ˢ (∅ : Set β) = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    ⊢ Eq (SProd.sprod s EmptyCollection.emptyCollection) EmptyCollection.emptyColl …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    x✝ : Prod α β
    ⊢ Iff (Membership.mem (SProd.sprod s EmptyCollection.emptyCollection) x✝) (Mem …
  -/
  exact iff_of_eq (and_false _)
  /-
    🎉 no goals
  -/


@[simp]
theorem empty_prod : (∅ : Set α) ×ˢ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    ⊢ Eq (SProd.sprod EmptyCollection.emptyCollection t) EmptyCollection.emptyColl …
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    t : Set β
    x✝ : Prod α β
    ⊢ Iff (Membership.mem (SProd.sprod EmptyCollection.emptyCollection t) x✝) (Mem …
  -/
  exact iff_of_eq (false_and _)
  /-
    🎉 no goals
  -/


@[simp, mfld_simps]
theorem univ_prod_univ : @univ α ×ˢ @univ β = univ := by
  /-
    α : Type u_1
    β : Type u_2
    ⊢ Eq (SProd.sprod Set.univ Set.univ) Set.univ
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    x✝ : Prod α β
    ⊢ Iff (Membership.mem (SProd.sprod Set.univ Set.univ) x✝) (Membership.mem Set. …
  -/
  exact iff_of_eq (true_and _)
  /-
    🎉 no goals
  -/


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             t : Set β
                                                                             ⊢ Eq (SProd.sprod Set.univ t) (Set.preimage Prod.snd t)
                                                                           -/
theorem univ_prod {t : Set β} : (univ : Set α) ×ˢ t = Prod.snd ⁻¹' t := by simp [prod_eq]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             s : Set α
                                                                             ⊢ Eq (SProd.sprod s Set.univ) (Set.preimage Prod.fst s)
                                                                           -/
theorem prod_univ {s : Set α} : s ×ˢ (univ : Set β) = Prod.fst ⁻¹' s := by simp [prod_eq]
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


@[simp] lemma prod_eq_univ [Nonempty α] [Nonempty β] : s ×ˢ t = univ ↔ s = univ ∧ t = univ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    inst✝¹ : Nonempty α
    inst✝ : Nonempty β
    ⊢ Iff (Eq (SProd.sprod s t) Set.univ) (And (Eq s Set.univ) (Eq t Set.univ))
  -/
  simp [eq_univ_iff_forall, forall_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_prod : ({a} : Set α) ×ˢ t = Prod.mk a '' t := by
  /-
    α : Type u_1
    β : Type u_2
    t : Set β
    a : α
    ⊢ Eq (SProd.sprod (Singleton.singleton a) t) (Set.image (Prod.mk a) t)
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    t : Set β
    a x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Singleton.singleton a) t) { fst := x, snd  …
  -/
  simp [and_left_comm, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_singleton : s ×ˢ ({b} : Set β) = (fun a => (a, b)) '' s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    b : β
    ⊢ Eq (SProd.sprod s (Singleton.singleton b)) (Set.image (fun a => { fst := a,  …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    b : β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Singleton.singleton b)) { fst := x, snd  …
  -/
  simp [and_left_comm, eq_comm]
  /-
    🎉 no goals
  -/


                                                                                   /-
                                                                                     α : Type u_1
                                                                                     β : Type u_2
                                                                                     a : α
                                                                                     b : β
                                                                                     ⊢ Eq (SProd.sprod (Singleton.singleton a) (Singleton.singleton b)) (Singleton. …
                                                                                   -/
theorem singleton_prod_singleton : ({a} : Set α) ×ˢ ({b} : Set β) = {(a, b)} := by simp
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[simp]
theorem union_prod : (s₁ ∪ s₂) ×ˢ t = s₁ ×ˢ t ∪ s₂ ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t : Set β
    ⊢ Eq (SProd.sprod (Union.union s₁ s₂) t) (Union.union (SProd.sprod s₁ t) (SPro …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t : Set β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Union.union s₁ s₂) t) { fst := x, snd := y …
  -/
  simp [or_and_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_union : s ×ˢ (t₁ ∪ t₂) = s ×ˢ t₁ ∪ s ×ˢ t₂ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t₁ t₂ : Set β
    ⊢ Eq (SProd.sprod s (Union.union t₁ t₂)) (Union.union (SProd.sprod s t₁) (SPro …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    t₁ t₂ : Set β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Union.union t₁ t₂)) { fst := x, snd := y …
  -/
  simp [and_or_left]
  /-
    🎉 no goals
  -/


theorem inter_prod : (s₁ ∩ s₂) ×ˢ t = s₁ ×ˢ t ∩ s₂ ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t : Set β
    ⊢ Eq (SProd.sprod (Inter.inter s₁ s₂) t) (Inter.inter (SProd.sprod s₁ t) (SPro …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t : Set β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Inter.inter s₁ s₂) t) { fst := x, snd := y …
  -/
  simp only [← and_and_right, mem_inter_iff, mem_prod]
  /-
    🎉 no goals
  -/


theorem prod_inter : s ×ˢ (t₁ ∩ t₂) = s ×ˢ t₁ ∩ s ×ˢ t₂ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t₁ t₂ : Set β
    ⊢ Eq (SProd.sprod s (Inter.inter t₁ t₂)) (Inter.inter (SProd.sprod s t₁) (SPro …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    t₁ t₂ : Set β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Inter.inter t₁ t₂)) { fst := x, snd := y …
  -/
  simp only [← and_and_left, mem_inter_iff, mem_prod]
  /-
    🎉 no goals
  -/


@[mfld_simps]
theorem prod_inter_prod : s₁ ×ˢ t₁ ∩ s₂ ×ˢ t₂ = (s₁ ∩ s₂) ×ˢ (t₁ ∩ t₂) := by
  /-
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    ⊢ Eq (Inter.inter (SProd.sprod s₁ t₁) (SProd.sprod s₂ t₂)) (SProd.sprod (Inter …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s₁ s₂ : Set α
    t₁ t₂ : Set β
    x : α
    y : β
    ⊢ Iff (Membership.mem (Inter.inter (SProd.sprod s₁ t₁) (SProd.sprod s₂ t₂)) {  …
  -/
  simp [and_assoc, and_left_comm]
  /-
    🎉 no goals
  -/


lemma compl_prod_eq_union {α β : Type*} (s : Set α) (t : Set β) :
    (s ×ˢ t)ᶜ = (sᶜ ×ˢ univ) ∪ (univ ×ˢ tᶜ) := by
  /-
    α : Type u_5
    β : Type u_6
    s : Set α
    t : Set β
    ⊢ Eq (HasCompl.compl (SProd.sprod s t)) (Union.union (SProd.sprod (HasCompl.co …
  -/
  ext p
  /-
    case h
    α : Type u_5
    β : Type u_6
    s : Set α
    t : Set β
    p : Prod α β
    ⊢ Iff (Membership.mem (HasCompl.compl (SProd.sprod s t)) p) (Membership.mem (U …
  -/
  simp only [mem_compl_iff, mem_prod, not_and, mem_union, mem_univ, and_true, true_and]
  /-
    case h
    α : Type u_5
    β : Type u_6
    s : Set α
    t : Set β
    p : Prod α β
    ⊢ Iff (Membership.mem s p.1 → Not (Membership.mem t p.2)) (Or (Not (Membership …
  -/
  constructor <;> intro h
    /-
      case h.mp
      α : Type u_5
      β : Type u_6
      s : Set α
      t : Set β
      p : Prod α β
      h : Membership.mem s p.1 → Not (Membership.mem t p.2)
      ⊢ Or (Not (Membership.mem s p.1)) (Not (Membership.mem t p.2))
    -/
  · by_cases fst_in_s : p.fst ∈ s
      /-
        case pos
        α : Type u_5
        β : Type u_6
        s : Set α
        t : Set β
        p : Prod α β
        h : Membership.mem s p.1 → Not (Membership.mem t p.2)
        fst_in_s : Membership.mem s p.1
        ⊢ Or (Not (Membership.mem s p.1)) (Not (Membership.mem t p.2))
      -/
    · exact Or.inr (h fst_in_s)
      /-
        🎉 no goals
      -/
      /-
        case neg
        α : Type u_5
        β : Type u_6
        s : Set α
        t : Set β
        p : Prod α β
        h : Membership.mem s p.1 → Not (Membership.mem t p.2)
        fst_in_s : Not (Membership.mem s p.1)
        ⊢ Or (Not (Membership.mem s p.1)) (Not (Membership.mem t p.2))
      -/
    · exact Or.inl fst_in_s
      /-
        🎉 no goals
      -/
    /-
      case h.mpr
      α : Type u_5
      β : Type u_6
      s : Set α
      t : Set β
      p : Prod α β
      h : Or (Not (Membership.mem s p.1)) (Not (Membership.mem t p.2))
      ⊢ Membership.mem s p.1 → Not (Membership.mem t p.2)
    -/
  · intro fst_in_s
    /-
      case h.mpr
      α : Type u_5
      β : Type u_6
      s : Set α
      t : Set β
      p : Prod α β
      h : Or (Not (Membership.mem s p.1)) (Not (Membership.mem t p.2))
      fst_in_s : Membership.mem s p.1
      ⊢ Not (Membership.mem t p.2)
    -/
    simpa only [fst_in_s, not_true, false_or] using h
    /-
      🎉 no goals
    -/


@[simp]
theorem disjoint_prod : Disjoint (s₁ ×ˢ t₁) (s₂ ×ˢ t₂) ↔ Disjoint s₁ s₂ ∨ Disjoint t₁ t₂ := by
  simp_rw [disjoint_left, mem_prod, not_and_or, Prod.forall, and_imp, ← @forall_or_right α, ←
    @forall_or_left β, ← @forall_or_right (_ ∈ s₁), ← @forall_or_left (_ ∈ t₁)]


theorem Disjoint.set_prod_left (hs : Disjoint s₁ s₂) (t₁ t₂ : Set β) :
    Disjoint (s₁ ×ˢ t₁) (s₂ ×ˢ t₂) :=
  disjoint_left.2 fun ⟨_a, _b⟩ ⟨ha₁, _⟩ ⟨ha₂, _⟩ => disjoint_left.1 hs ha₁ ha₂


theorem Disjoint.set_prod_right (ht : Disjoint t₁ t₂) (s₁ s₂ : Set α) :
    Disjoint (s₁ ×ˢ t₁) (s₂ ×ˢ t₂) :=
  disjoint_left.2 fun ⟨_a, _b⟩ ⟨_, hb₁⟩ ⟨_, hb₂⟩ => disjoint_left.1 ht hb₁ hb₂


theorem insert_prod : insert a s ×ˢ t = Prod.mk a '' t ∪ s ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ⊢ Eq (SProd.sprod (Insert.insert a s) t) (Union.union (Set.image (Prod.mk a) t …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Insert.insert a s) t) { fst := x, snd := y …
  -/
  simp +contextual [image, iff_def, or_imp]
  /-
    🎉 no goals
  -/


theorem prod_insert : s ×ˢ insert b t = (fun a => (a, b)) '' s ∪ s ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    ⊢ Eq (SProd.sprod s (Insert.insert b t)) (Union.union (Set.image (fun a => { f …
  -/
  ext ⟨x, y⟩
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745):
  -- was `simp +contextual [image, iff_def, or_imp, Imp.swap]`
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Insert.insert b t)) { fst := x, snd := y …
  -/
  simp only [mem_prod, mem_insert_iff, image, mem_union, mem_setOf_eq, Prod.mk.injEq]
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    x : α
    y : β
    ⊢ Iff (And (Membership.mem s x) (Or (Eq y b) (Membership.mem t y))) (Or (Exist …
  -/
  refine ⟨fun h => ?_, fun h => ?_⟩
    /-
      case h.mk.refine_1
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      b : β
      x : α
      y : β
      h : And (Membership.mem s x) (Or (Eq y b) (Membership.mem t y))
      ⊢ Or (Exists fun a => And (Membership.mem s a) (And (Eq a x) (Eq b y))) (And ( …
    -/
  · obtain ⟨hx, rfl|hy⟩ := h
      /-
        case h.mk.refine_1.intro.inl
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        x : α
        y : β
        hx : Membership.mem s x
        ⊢ Or (Exists fun a => And (Membership.mem s a) (And (Eq a x) (Eq y y))) (And ( …
      -/
    · exact Or.inl ⟨x, hx, rfl, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mk.refine_1.intro.inr
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        b : β
        x : α
        y : β
        hx : Membership.mem s x
        hy : Membership.mem t y
        ⊢ Or (Exists fun a => And (Membership.mem s a) (And (Eq a x) (Eq b y))) (And ( …
      -/
    · exact Or.inr ⟨hx, hy⟩
      /-
        🎉 no goals
      -/
    /-
      case h.mk.refine_2
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      b : β
      x : α
      y : β
      h : Or (Exists fun a => And (Membership.mem s a) (And (Eq a x) (Eq b y))) (And …
      ⊢ And (Membership.mem s x) (Or (Eq y b) (Membership.mem t y))
    -/
  · obtain ⟨x, hx, rfl, rfl⟩|⟨hx, hy⟩ := h
      /-
        case h.mk.refine_2.inl.intro.intro.intro
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        b : β
        x : α
        hx : Membership.mem s x
        ⊢ And (Membership.mem s x) (Or (Eq b b) (Membership.mem t b))
      -/
    · exact ⟨hx, Or.inl rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.mk.refine_2.inr.intro
        α : Type u_1
        β : Type u_2
        s : Set α
        t : Set β
        b : β
        x : α
        y : β
        hx : Membership.mem s x
        hy : Membership.mem t y
        ⊢ And (Membership.mem s x) (Or (Eq y b) (Membership.mem t y))
      -/
    · exact ⟨hx, Or.inr hy⟩
      /-
        🎉 no goals
      -/


theorem prod_preimage_eq {f : γ → α} {g : δ → β} :
    (f ⁻¹' s) ×ˢ (g ⁻¹' t) = (fun p : γ × δ => (f p.1, g p.2)) ⁻¹' s ×ˢ t :=
  rfl


theorem prod_preimage_left {f : γ → α} :
    (f ⁻¹' s) ×ˢ t = (fun p : γ × β => (f p.1, p.2)) ⁻¹' s ×ˢ t :=
  rfl


theorem prod_preimage_right {g : δ → β} :
    s ×ˢ (g ⁻¹' t) = (fun p : α × δ => (p.1, g p.2)) ⁻¹' s ×ˢ t :=
  rfl


theorem preimage_prod_map_prod (f : α → β) (g : γ → δ) (s : Set β) (t : Set δ) :
    Prod.map f g ⁻¹' s ×ˢ t = (f ⁻¹' s) ×ˢ (g ⁻¹' t) :=
  rfl


theorem mk_preimage_prod (f : γ → α) (g : γ → β) :
    (fun x => (f x, g x)) ⁻¹' s ×ˢ t = f ⁻¹' s ∩ g ⁻¹' t :=
  rfl


@[simp]
theorem mk_preimage_prod_left (hb : b ∈ t) : (fun a => (a, b)) ⁻¹' s ×ˢ t = s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    hb : Membership.mem t b
    ⊢ Eq (Set.preimage (fun a => { fst := a, snd := b }) (SProd.sprod s t)) s
  -/
  ext a
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    hb : Membership.mem t b
    a : α
    ⊢ Iff (Membership.mem (Set.preimage (fun a => { fst := a, snd := b }) (SProd.s …
  -/
  simp [hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_preimage_prod_right (ha : a ∈ s) : Prod.mk a ⁻¹' s ×ˢ t = t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ha : Membership.mem s a
    ⊢ Eq (Set.preimage (Prod.mk a) (SProd.sprod s t)) t
  -/
  ext b
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ha : Membership.mem s a
    b : β
    ⊢ Iff (Membership.mem (Set.preimage (Prod.mk a) (SProd.sprod s t)) b) (Members …
  -/
  simp [ha]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_preimage_prod_left_eq_empty (hb : b ∉ t) : (fun a => (a, b)) ⁻¹' s ×ˢ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    hb : Not (Membership.mem t b)
    ⊢ Eq (Set.preimage (fun a => { fst := a, snd := b }) (SProd.sprod s t)) EmptyC …
  -/
  ext a
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    hb : Not (Membership.mem t b)
    a : α
    ⊢ Iff (Membership.mem (Set.preimage (fun a => { fst := a, snd := b }) (SProd.s …
  -/
  simp [hb]
  /-
    🎉 no goals
  -/


@[simp]
theorem mk_preimage_prod_right_eq_empty (ha : a ∉ s) : Prod.mk a ⁻¹' s ×ˢ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Eq (Set.preimage (Prod.mk a) (SProd.sprod s t)) EmptyCollection.emptyCollect …
  -/
  ext b
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ha : Not (Membership.mem s a)
    b : β
    ⊢ Iff (Membership.mem (Set.preimage (Prod.mk a) (SProd.sprod s t)) b) (Members …
  -/
  simp [ha]
  /-
    🎉 no goals
  -/


theorem mk_preimage_prod_left_eq_if [DecidablePred (· ∈ t)] :
                                                                /-
                                                                  α : Type u_1
                                                                  β : Type u_2
                                                                  s : Set α
                                                                  t : Set β
                                                                  b : β
                                                                  inst✝ : DecidablePred fun x => Membership.mem t x
                                                                  ⊢ Eq (Set.preimage (fun a => { fst := a, snd := b }) (SProd.sprod s t)) (ite ( …
                                                                -/
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/
    (fun a => (a, b)) ⁻¹' s ×ˢ t = if b ∈ t then s else ∅ := by split_ifs with h <;> simp [h]
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem mk_preimage_prod_right_eq_if [DecidablePred (· ∈ s)] :
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          s : Set α
                                                          t : Set β
                                                          a : α
                                                          inst✝ : DecidablePred fun x => Membership.mem s x
                                                          ⊢ Eq (Set.preimage (Prod.mk a) (SProd.sprod s t)) (ite (Membership.mem s a) t  …
                                                        -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
    Prod.mk a ⁻¹' s ×ˢ t = if a ∈ s then t else ∅ := by split_ifs with h <;> simp [h]
                                                                             /-
                                                                               🎉 no goals
                                                                             -/


theorem mk_preimage_prod_left_fn_eq_if [DecidablePred (· ∈ t)] (f : γ → α) :
    (fun a => (f a, b)) ⁻¹' s ×ˢ t = if b ∈ t then f ⁻¹' s else ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Set α
    t : Set β
    b : β
    inst✝ : DecidablePred fun x => Membership.mem t x
    f : γ → α
    ⊢ Eq (Set.preimage (fun a => { fst := f a, snd := b }) (SProd.sprod s t)) (ite …
  -/
  rw [← mk_preimage_prod_left_eq_if, prod_preimage_left, preimage_preimage]
  /-
    🎉 no goals
  -/


theorem mk_preimage_prod_right_fn_eq_if [DecidablePred (· ∈ s)] (g : δ → β) :
    (fun b => (a, g b)) ⁻¹' s ×ˢ t = if a ∈ s then g ⁻¹' t else ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    δ : Type u_4
    s : Set α
    t : Set β
    a : α
    inst✝ : DecidablePred fun x => Membership.mem s x
    g : δ → β
    ⊢ Eq (Set.preimage (fun b => { fst := a, snd := g b }) (SProd.sprod s t)) (ite …
  -/
  rw [← mk_preimage_prod_right_eq_if, prod_preimage_right, preimage_preimage]
  /-
    🎉 no goals
  -/


@[simp]
theorem preimage_swap_prod (s : Set α) (t : Set β) : Prod.swap ⁻¹' s ×ˢ t = t ×ˢ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    ⊢ Eq (Set.preimage Prod.swap (SProd.sprod s t)) (SProd.sprod t s)
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    x : β
    y : α
    ⊢ Iff (Membership.mem (Set.preimage Prod.swap (SProd.sprod s t)) { fst := x, s …
  -/
  simp [and_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem image_swap_prod (s : Set α) (t : Set β) : Prod.swap '' s ×ˢ t = t ×ˢ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    ⊢ Eq (Set.image Prod.swap (SProd.sprod s t)) (SProd.sprod t s)
  -/
  rw [image_swap_eq_preimage_swap, preimage_swap_prod]
  /-
    🎉 no goals
  -/


theorem prod_image_image_eq {m₁ : α → γ} {m₂ : β → δ} :
    (m₁ '' s) ×ˢ (m₂ '' t) = (fun p : α × β => (m₁ p.1, m₂ p.2)) '' s ×ˢ t :=
  ext <| by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      s : Set α
      t : Set β
      m₁ : α → γ
      m₂ : β → δ
      ⊢ ∀ (x : Prod γ δ), Iff (Membership.mem (SProd.sprod (Set.image m₁ s) (Set.ima …
    -/
    simp [-exists_and_right, exists_and_right.symm, and_left_comm, and_assoc, and_comm]
    /-
      🎉 no goals
    -/


theorem prod_range_range_eq {m₁ : α → γ} {m₂ : β → δ} :
    range m₁ ×ˢ range m₂ = range fun p : α × β => (m₁ p.1, m₂ p.2) :=
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              δ : Type u_4
              m₁ : α → γ
              m₂ : β → δ
              ⊢ ∀ (x : Prod γ δ), Iff (Membership.mem (SProd.sprod (Set.range m₁) (Set.range …
            -/
  ext <| by simp [range]
            /-
              🎉 no goals
            -/


@[simp, mfld_simps]
theorem range_prod_map {m₁ : α → γ} {m₂ : β → δ} : range (Prod.map m₁ m₂) = range m₁ ×ˢ range m₂ :=
  prod_range_range_eq.symm


theorem prod_range_univ_eq {m₁ : α → γ} :
    range m₁ ×ˢ (univ : Set β) = range fun p : α × β => (m₁ p.1, p.2) :=
            /-
              α : Type u_1
              β : Type u_2
              γ : Type u_3
              m₁ : α → γ
              ⊢ ∀ (x : Prod γ β), Iff (Membership.mem (SProd.sprod (Set.range m₁) Set.univ)  …
            -/
  ext <| by simp [range]
            /-
              🎉 no goals
            -/


theorem prod_univ_range_eq {m₂ : β → δ} :
    (univ : Set α) ×ˢ range m₂ = range fun p : α × β => (p.1, m₂ p.2) :=
            /-
              α : Type u_1
              β : Type u_2
              δ : Type u_4
              m₂ : β → δ
              ⊢ ∀ (x : Prod α δ), Iff (Membership.mem (SProd.sprod Set.univ (Set.range m₂))  …
            -/
  ext <| by simp [range]
            /-
              🎉 no goals
            -/


theorem range_pair_subset (f : α → β) (g : α → γ) :
    (range fun x => (f x, g x)) ⊆ range f ×ˢ range g := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    ⊢ HasSubset.Subset (Set.range fun x => { fst := f x, snd := g x }) (SProd.spro …
  -/
  have : (fun x => (f x, g x)) = Prod.map f g ∘ fun x => (x, x) := funext fun x => rfl
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    this : Eq (fun x => { fst := f x, snd := g x }) (Function.comp (Prod.map f g)  …
    ⊢ HasSubset.Subset (Set.range fun x => { fst := f x, snd := g x }) (SProd.spro …
  -/
  rw [this, ← range_prod_map]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    this : Eq (fun x => { fst := f x, snd := g x }) (Function.comp (Prod.map f g)  …
    ⊢ HasSubset.Subset (Set.range (Function.comp (Prod.map f g) fun x => { fst :=  …
  -/
  apply range_comp_subset_range
  /-
    🎉 no goals
  -/


theorem Nonempty.prod : s.Nonempty → t.Nonempty → (s ×ˢ t).Nonempty := fun ⟨x, hx⟩ ⟨y, hy⟩ =>
  ⟨(x, y), ⟨hx, hy⟩⟩


theorem Nonempty.fst : (s ×ˢ t).Nonempty → s.Nonempty := fun ⟨x, hx⟩ => ⟨x.1, hx.1⟩


theorem Nonempty.snd : (s ×ˢ t).Nonempty → t.Nonempty := fun ⟨x, hx⟩ => ⟨x.2, hx.2⟩


@[simp]
theorem prod_nonempty_iff : (s ×ˢ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  ⟨fun h => ⟨h.fst, h.snd⟩, fun h => h.1.prod h.2⟩


@[simp]
theorem prod_eq_empty_iff : s ×ˢ t = ∅ ↔ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    ⊢ Iff (Eq (SProd.sprod s t) EmptyCollection.emptyCollection) (Or (Eq s EmptyCo …
  -/
  simp only [not_nonempty_iff_eq_empty.symm, prod_nonempty_iff, not_and_or]
  /-
    🎉 no goals
  -/


theorem prod_sub_preimage_iff {W : Set γ} {f : α × β → γ} :
                                                                 /-
                                                                   α : Type u_1
                                                                   β : Type u_2
                                                                   γ : Type u_3
                                                                   s : Set α
                                                                   t : Set β
                                                                   W : Set γ
                                                                   f : Prod α β → γ
                                                                   ⊢ Iff (HasSubset.Subset (SProd.sprod s t) (Set.preimage f W)) (∀ (a : α) (b :  …
                                                                 -/
    s ×ˢ t ⊆ f ⁻¹' W ↔ ∀ a b, a ∈ s → b ∈ t → f (a, b) ∈ W := by simp [subset_def]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


theorem image_prod_mk_subset_prod {f : α → β} {g : α → γ} {s : Set α} :
    (fun x => (f x, g x)) '' s ⊆ (f '' s) ×ˢ (g '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    s : Set α
    ⊢ HasSubset.Subset (Set.image (fun x => { fst := f x, snd := g x }) s) (SProd. …
  -/
  rintro _ ⟨x, hx, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    g : α → γ
    s : Set α
    x : α
    hx : Membership.mem s x
    ⊢ Membership.mem (SProd.sprod (Set.image f s) (Set.image g s)) ((fun x => { fs …
  -/
  exact mk_mem_prod (mem_image_of_mem f hx) (mem_image_of_mem g hx)
  /-
    🎉 no goals
  -/


theorem image_prod_mk_subset_prod_left (hb : b ∈ t) : (fun a => (a, b)) '' s ⊆ s ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    hb : Membership.mem t b
    ⊢ HasSubset.Subset (Set.image (fun a => { fst := a, snd := b }) s) (SProd.spro …
  -/
  rintro _ ⟨a, ha, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    b : β
    hb : Membership.mem t b
    a : α
    ha : Membership.mem s a
    ⊢ Membership.mem (SProd.sprod s t) ((fun a => { fst := a, snd := b }) a)
  -/
  exact ⟨ha, hb⟩
  /-
    🎉 no goals
  -/


theorem image_prod_mk_subset_prod_right (ha : a ∈ s) : Prod.mk a '' t ⊆ s ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ha : Membership.mem s a
    ⊢ HasSubset.Subset (Set.image (Prod.mk a) t) (SProd.sprod s t)
  -/
  rintro _ ⟨b, hb, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    s : Set α
    t : Set β
    a : α
    ha : Membership.mem s a
    b : β
    hb : Membership.mem t b
    ⊢ Membership.mem (SProd.sprod s t) { fst := a, snd := b }
  -/
  exact ⟨ha, hb⟩
  /-
    🎉 no goals
  -/


theorem prod_subset_preimage_fst (s : Set α) (t : Set β) : s ×ˢ t ⊆ Prod.fst ⁻¹' s :=
  inter_subset_left


theorem fst_image_prod_subset (s : Set α) (t : Set β) : Prod.fst '' s ×ˢ t ⊆ s :=
  image_subset_iff.2 <| prod_subset_preimage_fst s t


theorem fst_image_prod (s : Set β) {t : Set α} (ht : t.Nonempty) : Prod.fst '' s ×ˢ t = s :=
  (fst_image_prod_subset _ _).antisymm fun y hy =>
    let ⟨x, hx⟩ := ht
    ⟨(y, x), ⟨hy, hx⟩, rfl⟩


lemma mapsTo_fst_prod {s : Set α} {t : Set β} : MapsTo Prod.fst (s ×ˢ t) s :=
  fun _ hx ↦ (mem_prod.1 hx).1


theorem prod_subset_preimage_snd (s : Set α) (t : Set β) : s ×ˢ t ⊆ Prod.snd ⁻¹' t :=
  inter_subset_right


theorem snd_image_prod_subset (s : Set α) (t : Set β) : Prod.snd '' s ×ˢ t ⊆ t :=
  image_subset_iff.2 <| prod_subset_preimage_snd s t


theorem snd_image_prod {s : Set α} (hs : s.Nonempty) (t : Set β) : Prod.snd '' s ×ˢ t = t :=
  (snd_image_prod_subset _ _).antisymm fun y y_in =>
    let ⟨x, x_in⟩ := hs
    ⟨(x, y), ⟨x_in, y_in⟩, rfl⟩


lemma mapsTo_snd_prod {s : Set α} {t : Set β} : MapsTo Prod.snd (s ×ˢ t) t :=
  fun _ hx ↦ (mem_prod.1 hx).2


theorem prod_diff_prod : s ×ˢ t \ s₁ ×ˢ t₁ = s ×ˢ (t \ t₁) ∪ (s \ s₁) ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    ⊢ Eq (SDiff.sdiff (SProd.sprod s t) (SProd.sprod s₁ t₁)) (Union.union (SProd.s …
  -/
  ext x
  /-
    case h
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    x : Prod α β
    ⊢ Iff (Membership.mem (SDiff.sdiff (SProd.sprod s t) (SProd.sprod s₁ t₁)) x) ( …
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
  by_cases h₁ : x.1 ∈ s₁ <;> by_cases h₂ : x.2 ∈ t₁ <;> simp [*]
                                                        /-
                                                          🎉 no goals
                                                        -/


/-- A product set is included in a product set if and only factors are included, or a factor of the
first set is empty. -/
theorem prod_subset_prod_iff : s ×ˢ t ⊆ s₁ ×ˢ t₁ ↔ s ⊆ s₁ ∧ t ⊆ t₁ ∨ s = ∅ ∨ t = ∅ := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    ⊢ Iff (HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)) (Or (And (HasSu …
  -/
  rcases (s ×ˢ t).eq_empty_or_nonempty with h | h
    /-
      case inl
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : Eq (SProd.sprod s t) EmptyCollection.emptyCollection
      ⊢ Iff (HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)) (Or (And (HasSu …
    -/
  · simp [h, prod_eq_empty_iff.1 h]
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    h : (SProd.sprod s t).Nonempty
    ⊢ Iff (HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)) (Or (And (HasSu …
  -/
  have st : s.Nonempty ∧ t.Nonempty := by rwa [prod_nonempty_iff] at h
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    h : (SProd.sprod s t).Nonempty
    st : And s.Nonempty t.Nonempty
    ⊢ Iff (HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)) (Or (And (HasSu …
  -/
  refine ⟨fun H => Or.inl ⟨?_, ?_⟩, ?_⟩
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      H : HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)
      ⊢ HasSubset.Subset s s₁
    -/
  · have := image_subset (Prod.fst : α × β → α) H
    /-
      case inr.refine_1
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      H : HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)
      this : HasSubset.Subset (Set.image Prod.fst (SProd.sprod s t)) (Set.image Prod …
      ⊢ HasSubset.Subset s s₁
    -/
    rwa [fst_image_prod _ st.2, fst_image_prod _ (h.mono H).snd] at this
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_2
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      H : HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)
      ⊢ HasSubset.Subset t t₁
    -/
  · have := image_subset (Prod.snd : α × β → β) H
    /-
      case inr.refine_2
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      H : HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)
      this : HasSubset.Subset (Set.image Prod.snd (SProd.sprod s t)) (Set.image Prod …
      ⊢ HasSubset.Subset t t₁
    -/
    rwa [snd_image_prod st.1, snd_image_prod (h.mono H).fst] at this
    /-
      🎉 no goals
    -/
    /-
      case inr.refine_3
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      ⊢ Or (And (HasSubset.Subset s s₁) (HasSubset.Subset t t₁)) (Or (Eq s EmptyColl …
    -/
  · intro H
    /-
      case inr.refine_3
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      H : Or (And (HasSubset.Subset s s₁) (HasSubset.Subset t t₁)) (Or (Eq s EmptyCo …
      ⊢ HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)
    -/
    simp only [st.1.ne_empty, st.2.ne_empty, or_false] at H
    /-
      case inr.refine_3
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      st : And s.Nonempty t.Nonempty
      H : And (HasSubset.Subset s s₁) (HasSubset.Subset t t₁)
      ⊢ HasSubset.Subset (SProd.sprod s t) (SProd.sprod s₁ t₁)
    -/
    exact prod_mono H.1 H.2
    /-
      🎉 no goals
    -/


theorem prod_eq_prod_iff_of_nonempty (h : (s ×ˢ t).Nonempty) :
    s ×ˢ t = s₁ ×ˢ t₁ ↔ s = s₁ ∧ t = t₁ := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    h : (SProd.sprod s t).Nonempty
    ⊢ Iff (Eq (SProd.sprod s t) (SProd.sprod s₁ t₁)) (And (Eq s s₁) (Eq t t₁))
  -/
  constructor
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      ⊢ Eq (SProd.sprod s t) (SProd.sprod s₁ t₁) → And (Eq s s₁) (Eq t t₁)
    -/
  · intro heq
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      heq : Eq (SProd.sprod s t) (SProd.sprod s₁ t₁)
      ⊢ And (Eq s s₁) (Eq t t₁)
    -/
    have h₁ : (s₁ ×ˢ t₁ : Set _).Nonempty := by rwa [← heq]
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      heq : Eq (SProd.sprod s t) (SProd.sprod s₁ t₁)
      h₁ : (SProd.sprod s₁ t₁).Nonempty
      ⊢ And (Eq s s₁) (Eq t t₁)
    -/
    rw [prod_nonempty_iff] at h h₁
    rw [← fst_image_prod s h.2, ← fst_image_prod s₁ h₁.2, heq, eq_self_iff_true, true_and, ←
      snd_image_prod h.1 t, ← snd_image_prod h₁.1 t₁, heq]
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : (SProd.sprod s t).Nonempty
      ⊢ And (Eq s s₁) (Eq t t₁) → Eq (SProd.sprod s t) (SProd.sprod s₁ t₁)
    -/
  · rintro ⟨rfl, rfl⟩
    /-
      case mpr.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      h : (SProd.sprod s t).Nonempty
      ⊢ Eq (SProd.sprod s t) (SProd.sprod s t)
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem prod_eq_prod_iff :
    s ×ˢ t = s₁ ×ˢ t₁ ↔ s = s₁ ∧ t = t₁ ∨ (s = ∅ ∨ t = ∅) ∧ (s₁ = ∅ ∨ t₁ = ∅) := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    ⊢ Iff (Eq (SProd.sprod s t) (SProd.sprod s₁ t₁)) (Or (And (Eq s s₁) (Eq t t₁)) …
  -/
  symm
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    ⊢ Iff (Or (And (Eq s s₁) (Eq t t₁)) (And (Or (Eq s EmptyCollection.emptyCollec …
  -/
  rcases eq_empty_or_nonempty (s ×ˢ t) with h | h
  · simp_rw [h, @eq_comm _ ∅, prod_eq_empty_iff, prod_eq_empty_iff.mp h, true_and,
      or_iff_right_iff_imp]
    /-
      case inl
      α : Type u_1
      β : Type u_2
      s s₁ : Set α
      t t₁ : Set β
      h : Eq (SProd.sprod s t) EmptyCollection.emptyCollection
      ⊢ And (Eq s s₁) (Eq t t₁) → Or (Eq s₁ EmptyCollection.emptyCollection) (Eq t₁  …
    -/
    rintro ⟨rfl, rfl⟩
    /-
      case inl.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      t : Set β
      h : Eq (SProd.sprod s t) EmptyCollection.emptyCollection
      ⊢ Or (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.emptyCollect …
    -/
    exact prod_eq_empty_iff.mp h
    /-
      🎉 no goals
    -/
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    h : (SProd.sprod s t).Nonempty
    ⊢ Iff (Or (And (Eq s s₁) (Eq t t₁)) (And (Or (Eq s EmptyCollection.emptyCollec …
  -/
  rw [prod_eq_prod_iff_of_nonempty h]
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    h : (SProd.sprod s t).Nonempty
    ⊢ Iff (Or (And (Eq s s₁) (Eq t t₁)) (And (Or (Eq s EmptyCollection.emptyCollec …
  -/
  rw [nonempty_iff_ne_empty, Ne, prod_eq_empty_iff] at h
  /-
    case inr
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t t₁ : Set β
    h : Not (Or (Eq s EmptyCollection.emptyCollection) (Eq t EmptyCollection.empty …
    ⊢ Iff (Or (And (Eq s s₁) (Eq t t₁)) (And (Or (Eq s EmptyCollection.emptyCollec …
  -/
  simp_rw [h, false_and, or_false]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_eq_iff_eq (ht : t.Nonempty) : s ×ˢ t = s₁ ×ˢ t ↔ s = s₁ := by
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    ht : t.Nonempty
    ⊢ Iff (Eq (SProd.sprod s t) (SProd.sprod s₁ t)) (Eq s s₁)
  -/
  simp_rw [prod_eq_prod_iff, ht.ne_empty, and_true, or_iff_left_iff_imp, or_false]
  /-
    α : Type u_1
    β : Type u_2
    s s₁ : Set α
    t : Set β
    ht : t.Nonempty
    ⊢ And (Eq s EmptyCollection.emptyCollection) (Eq s₁ EmptyCollection.emptyColle …
  -/
  rintro ⟨rfl, rfl⟩
  /-
    case intro
    α : Type u_1
    β : Type u_2
    t : Set β
    ht : t.Nonempty
    ⊢ Eq EmptyCollection.emptyCollection EmptyCollection.emptyCollection
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem subset_prod {s : Set (α × β)} : s ⊆ (Prod.fst '' s) ×ˢ (Prod.snd '' s) :=
  fun _ hp ↦ mem_prod.2 ⟨mem_image_of_mem _ hp, mem_image_of_mem _ hp⟩


theorem _root_.Monotone.set_prod (hf : Monotone f) (hg : Monotone g) :
    Monotone fun x => f x ×ˢ g x :=
  fun _ _ h => prod_mono (hf h) (hg h)


theorem _root_.Antitone.set_prod (hf : Antitone f) (hg : Antitone g) :
    Antitone fun x => f x ×ˢ g x :=
  fun _ _ h => prod_mono (hf h) (hg h)


theorem _root_.MonotoneOn.set_prod (hf : MonotoneOn f s) (hg : MonotoneOn g s) :
    MonotoneOn (fun x => f x ×ˢ g x) s := fun _ ha _ hb h => prod_mono (hf ha hb h) (hg ha hb h)


theorem _root_.AntitoneOn.set_prod (hf : AntitoneOn f s) (hg : AntitoneOn g s) :
    AntitoneOn (fun x => f x ×ˢ g x) s := fun _ ha _ hb h => prod_mono (hf ha hb h) (hg ha hb h)


lemma diagonal_nonempty [Nonempty α] : (diagonal α).Nonempty :=
  Nonempty.elim ‹_› fun x => ⟨_, mem_diagonal x⟩


instance decidableMemDiagonal [h : DecidableEq α] (x : α × α) : Decidable (x ∈ diagonal α) :=
  h x.1 x.2


theorem preimage_coe_coe_diagonal (s : Set α) :
    Prod.map (fun x : s => (x : α)) (fun x : s => (x : α)) ⁻¹' diagonal α = diagonal s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.preimage (Prod.map (fun x => ↑x) fun x => ↑x) (Set.diagonal α)) (Set …
  -/
  ext ⟨⟨x, hx⟩, ⟨y, hy⟩⟩
  /-
    case h.mk.mk.mk
    α : Type u_1
    s : Set α
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    ⊢ Iff (Membership.mem (Set.preimage (Prod.map (fun x => ↑x) fun x => ↑x) (Set. …
  -/
  simp [Set.diagonal]
  /-
    🎉 no goals
  -/


@[simp]
theorem range_diag : (range fun x => (x, x)) = diagonal α := by
  /-
    α : Type u_1
    ⊢ Eq (Set.range fun x => { fst := x, snd := x }) (Set.diagonal α)
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    x y : α
    ⊢ Iff (Membership.mem (Set.range fun x => { fst := x, snd := x }) { fst := x,  …
  -/
  simp [diagonal, eq_comm]
  /-
    🎉 no goals
  -/


theorem diagonal_subset_iff {s} : diagonal α ⊆ s ↔ ∀ x, (x, x) ∈ s := by
  /-
    α : Type u_1
    s : Set (Prod α α)
    ⊢ Iff (HasSubset.Subset (Set.diagonal α) s) (∀ (x : α), Membership.mem s { fst …
  -/
  rw [← range_diag, range_subset_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem prod_subset_compl_diagonal_iff_disjoint : s ×ˢ t ⊆ (diagonal α)ᶜ ↔ Disjoint s t :=
  prod_subset_iff.trans disjoint_iff_forall_ne.symm


@[simp]
theorem diag_preimage_prod (s t : Set α) : (fun x => (x, x)) ⁻¹' s ×ˢ t = s ∩ t :=
  rfl


theorem diag_preimage_prod_self (s : Set α) : (fun x => (x, x)) ⁻¹' s ×ˢ s = s :=
  inter_self s


theorem diag_image (s : Set α) : (fun x => (x, x)) '' s = diagonal α ∩ s ×ˢ s := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Eq (Set.image (fun x => { fst := x, snd := x }) s) (Inter.inter (Set.diagona …
  -/
  rw [← range_diag, ← image_preimage_eq_range_inter, diag_preimage_prod_self]
  /-
    🎉 no goals
  -/


theorem diagonal_eq_univ_iff : diagonal α = univ ↔ Subsingleton α := by
  /-
    α : Type u_1
    ⊢ Iff (Eq (Set.diagonal α) Set.univ) (Subsingleton α)
  -/
  simp only [subsingleton_iff, eq_univ_iff_forall, Prod.forall, mem_diagonal_iff]
  /-
    🎉 no goals
  -/


theorem diagonal_eq_univ [Subsingleton α] : diagonal α = univ := diagonal_eq_univ_iff.2 ‹_›


/-- A function is `Function.const α a` for some `a` if and only if `∀ x y, f x = f y`. -/
theorem range_const_eq_diagonal {α β : Type*} [hβ : Nonempty β] :
    range (const α) = {f : α → β | ∀ x y, f x = f y} := by
  /-
    α : Type u_1
    β : Type u_2
    hβ : Nonempty β
    ⊢ Eq (Set.range (Function.const α)) (setOf fun f => ∀ (x y : α), Eq (f x) (f y))
  -/
  refine (range_eq_iff _ _).mpr ⟨fun _ _ _ ↦ rfl, fun f hf ↦ ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    hβ : Nonempty β
    f : α → β
    hf : Membership.mem (setOf fun f => ∀ (x y : α), Eq (f x) (f y)) f
    ⊢ Exists fun a => Eq (Function.const α a) f
  -/
  rcases isEmpty_or_nonempty α with h|⟨⟨a⟩⟩
    /-
      case inl
      α : Type u_1
      β : Type u_2
      hβ : Nonempty β
      f : α → β
      hf : Membership.mem (setOf fun f => ∀ (x y : α), Eq (f x) (f y)) f
      h : IsEmpty α
      ⊢ Exists fun a => Eq (Function.const α a) f
    -/
  · exact hβ.elim fun b ↦ ⟨b, Subsingleton.elim _ _⟩
    /-
      🎉 no goals
    -/
    /-
      case inr.intro
      α : Type u_1
      β : Type u_2
      hβ : Nonempty β
      f : α → β
      hf : Membership.mem (setOf fun f => ∀ (x y : α), Eq (f x) (f y)) f
      a : α
      ⊢ Exists fun a => Eq (Function.const α a) f
    -/
  · exact ⟨f a, funext fun x ↦ hf _ _⟩
    /-
      🎉 no goals
    -/


/-- The fiber product $X \times_Y Z$. -/
abbrev Function.Pullback (f : X → Y) (g : Z → Y) := {p : X × Z // f p.1 = g p.2}


/-- The fiber product $X \times_Y X$. -/
abbrev Function.PullbackSelf (f : X → Y) := f.Pullback f


/-- The projection from the fiber product to the first factor. -/
def Function.Pullback.fst {f : X → Y} {g : Z → Y} (p : f.Pullback g) : X := p.val.1


/-- The projection from the fiber product to the second factor. -/
def Function.Pullback.snd {f : X → Y} {g : Z → Y} (p : f.Pullback g) : Z := p.val.2


open Function.Pullback in
lemma Function.pullback_comm_sq (f : X → Y) (g : Z → Y) :
    f ∘ @fst X Y Z f g = g ∘ @snd X Y Z f g := funext fun p ↦ p.2


/-- The diagonal map $\Delta: X \to X \times_Y X$. -/
def toPullbackDiag (f : X → Y) (x : X) : f.Pullback f := ⟨(x, x), rfl⟩


/-- The diagonal $\Delta(X) \subseteq X \times_Y X$. -/
def Function.pullbackDiagonal (f : X → Y) : Set (f.Pullback f) := {p | p.fst = p.snd}


/-- Three functions between the three pairs of spaces $X_i, Y_i, Z_i$ that are compatible
  induce a function $X_1 \times_{Y_1} Z_1 \to X_2 \times_{Y_2} Z_2$. -/
def Function.mapPullback {X₁ X₂ Y₁ Y₂ Z₁ Z₂}
    {f₁ : X₁ → Y₁} {g₁ : Z₁ → Y₁} {f₂ : X₂ → Y₂} {g₂ : Z₂ → Y₂}
    (mapX : X₁ → X₂) (mapY : Y₁ → Y₂) (mapZ : Z₁ → Z₂)
    (commX : f₂ ∘ mapX = mapY ∘ f₁) (commZ : g₂ ∘ mapZ = mapY ∘ g₁)
    (p : f₁.Pullback g₁) : f₂.Pullback g₂ :=
  ⟨(mapX p.fst, mapZ p.snd),
    (congr_fun commX _).trans <| (congr_arg mapY p.2).trans <| congr_fun commZ.symm _⟩


open Function.Pullback in
/-- The projection $(X \times_Y Z) \times_Z (X \times_Y Z) \to X \times_Y X$. -/
def Function.PullbackSelf.map_fst {f : X → Y} {g : Z → Y} :
    (@snd X Y Z f g).PullbackSelf → f.PullbackSelf :=
  mapPullback fst g fst (pullback_comm_sq f g) (pullback_comm_sq f g)


open Function.Pullback in
/-- The projection $(X \times_Y Z) \times_X (X \times_Y Z) \to Z \times_Y Z$. -/
def Function.PullbackSelf.map_snd {f : X → Y} {g : Z → Y} :
    (@fst X Y Z f g).PullbackSelf → g.PullbackSelf :=
  mapPullback snd f snd (pullback_comm_sq f g).symm (pullback_comm_sq f g).symm


theorem preimage_map_fst_pullbackDiagonal {f : X → Y} {g : Z → Y} :
    @map_fst X Y Z f g ⁻¹' pullbackDiagonal f = pullbackDiagonal (@snd X Y Z f g) := by
  /-
    X : Type u_2
    Y : Sort u_3
    Z : Type u_1
    f : X → Y
    g : Z → Y
    ⊢ Eq (Set.preimage Function.PullbackSelf.map_fst (Function.pullbackDiagonal f) …
  -/
  ext ⟨⟨p₁, p₂⟩, he⟩
  /-
    case h.mk.mk
    X : Type u_2
    Y : Sort u_3
    Z : Type u_1
    f : X → Y
    g : Z → Y
    p₁ p₂ : Function.Pullback f g
    he : Eq { fst := p₁, snd := p₂ }.1.snd { fst := p₁, snd := p₂ }.2.snd
    ⊢ Iff (Membership.mem (Set.preimage Function.PullbackSelf.map_fst (Function.pu …
  -/
  simp_rw [pullbackDiagonal, mem_setOf, Subtype.ext_iff, Prod.ext_iff]
  /-
    case h.mk.mk
    X : Type u_2
    Y : Sort u_3
    Z : Type u_1
    f : X → Y
    g : Z → Y
    p₁ p₂ : Function.Pullback f g
    he : Eq { fst := p₁, snd := p₂ }.1.snd { fst := p₁, snd := p₂ }.2.snd
    ⊢ Iff (Membership.mem (Set.preimage Function.PullbackSelf.map_fst (setOf fun p …
  -/
  exact (and_iff_left he).symm
  /-
    🎉 no goals
  -/


theorem Function.Injective.preimage_pullbackDiagonal {f : X → Y} {g : Z → X} (inj : g.Injective) :
                           /-
                             X : Type ?u.83620
                             Y : Sort ?u.83594
                             Z : Type ?u.83621
                             f : X → Y
                             g : Z → X
                             inj : Function.Injective g
                             ⊢ Eq (Function.comp f g) (Function.comp id (Function.comp f g))
                           -/
                           /-
                             🎉 no goals
                           -/
    mapPullback g id g (by rfl) (by rfl) ⁻¹' pullbackDiagonal f = pullbackDiagonal (f ∘ g) :=
                                    /-
                                      🎉 no goals
                                    -/
  ext fun _ ↦ inj.eq_iff


theorem image_toPullbackDiag (f : X → Y) (s : Set X) :
    toPullbackDiag f '' s = pullbackDiagonal f ∩ Subtype.val ⁻¹' s ×ˢ s := by
  /-
    X : Type u_1
    Y : Sort u_2
    f : X → Y
    s : Set X
    ⊢ Eq (Set.image (toPullbackDiag f) s) (Inter.inter (Function.pullbackDiagonal  …
  -/
  ext x
  /-
    case h
    X : Type u_1
    Y : Sort u_2
    f : X → Y
    s : Set X
    x : Function.Pullback f f
    ⊢ Iff (Membership.mem (Set.image (toPullbackDiag f) s) x) (Membership.mem (Int …
  -/
  constructor
    /-
      case h.mp
      X : Type u_1
      Y : Sort u_2
      f : X → Y
      s : Set X
      x : Function.Pullback f f
      ⊢ Membership.mem (Set.image (toPullbackDiag f) s) x → Membership.mem (Inter.in …
    -/
  · rintro ⟨x, hx, rfl⟩
    /-
      case h.mp.intro.intro
      X : Type u_1
      Y : Sort u_2
      f : X → Y
      s : Set X
      x : X
      hx : Membership.mem s x
      ⊢ Membership.mem (Inter.inter (Function.pullbackDiagonal f) (Set.preimage Subt …
    -/
    exact ⟨rfl, hx, hx⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      X : Type u_1
      Y : Sort u_2
      f : X → Y
      s : Set X
      x : Function.Pullback f f
      ⊢ Membership.mem (Inter.inter (Function.pullbackDiagonal f) (Set.preimage Subt …
    -/
  · obtain ⟨⟨x, y⟩, h⟩ := x
    /-
      case h.mpr.mk.mk
      X : Type u_1
      Y : Sort u_2
      f : X → Y
      s : Set X
      x y : X
      h : Eq (f { fst := x, snd := y }.1) (f { fst := x, snd := y }.2)
      ⊢ Membership.mem (Inter.inter (Function.pullbackDiagonal f) (Set.preimage Subt …
    -/
    rintro ⟨rfl : x = y, h2x⟩
    /-
      case h.mpr.mk.mk.intro
      X : Type u_1
      Y : Sort u_2
      f : X → Y
      s : Set X
      x : X
      h : Eq (f { fst := x, snd := x }.1) (f { fst := x, snd := x }.2)
      h2x : Membership.mem (Set.preimage Subtype.val (SProd.sprod s s)) ⟨{ fst := x, …
      ⊢ Membership.mem (Set.image (toPullbackDiag f) s) ⟨{ fst := x, snd := x }, h⟩
    -/
    exact mem_image_of_mem _ h2x.1
    /-
      🎉 no goals
    -/


theorem range_toPullbackDiag (f : X → Y) : range (toPullbackDiag f) = pullbackDiagonal f := by
  /-
    X : Type u_1
    Y : Sort u_2
    f : X → Y
    ⊢ Eq (Set.range (toPullbackDiag f)) (Function.pullbackDiagonal f)
  -/
  rw [← image_univ, image_toPullbackDiag, univ_prod_univ, preimage_univ, inter_univ]
  /-
    🎉 no goals
  -/


theorem injective_toPullbackDiag (f : X → Y) : (toPullbackDiag f).Injective :=
  fun _ _ h ↦ congr_arg Prod.fst (congr_arg Subtype.val h)


theorem offDiag_mono : Monotone (offDiag : Set α → Set (α × α)) := fun _ _ h _ =>
  And.imp (@h _) <| And.imp_left <| @h _


@[simp]
theorem offDiag_nonempty : s.offDiag.Nonempty ↔ s.Nontrivial := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff s.offDiag.Nonempty s.Nontrivial
  -/
  simp [offDiag, Set.Nonempty, Set.Nontrivial]
  /-
    🎉 no goals
  -/


@[simp]
theorem offDiag_eq_empty : s.offDiag = ∅ ↔ s.Subsingleton := by
  /-
    α : Type u_1
    s : Set α
    ⊢ Iff (Eq s.offDiag EmptyCollection.emptyCollection) s.Subsingleton
  -/
  rw [← not_nonempty_iff_eq_empty, ← not_nontrivial_iff, offDiag_nonempty.not]
  /-
    🎉 no goals
  -/


alias ⟨_, Nontrivial.offDiag_nonempty⟩ := offDiag_nonempty


alias ⟨_, Subsingleton.offDiag_eq_empty⟩ := offDiag_nonempty


theorem offDiag_subset_prod : s.offDiag ⊆ s ×ˢ s := fun _ hx => ⟨hx.1, hx.2.1⟩


theorem offDiag_eq_sep_prod : s.offDiag = { x ∈ s ×ˢ s | x.1 ≠ x.2 } :=
  ext fun _ => and_assoc.symm


@[simp]
                                                      /-
                                                        α : Type u_1
                                                        ⊢ Eq EmptyCollection.emptyCollection.offDiag EmptyCollection.emptyCollection
                                                      -/
theorem offDiag_empty : (∅ : Set α).offDiag = ∅ := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


@[simp]
                                                                    /-
                                                                      α : Type u_1
                                                                      a : α
                                                                      ⊢ Eq (Singleton.singleton a).offDiag EmptyCollection.emptyCollection
                                                                    -/
theorem offDiag_singleton (a : α) : ({a} : Set α).offDiag = ∅ := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem offDiag_univ : (univ : Set α).offDiag = (diagonal α)ᶜ :=
            /-
              α : Type u_1
              ⊢ ∀ (x : Prod α α), Iff (Membership.mem Set.univ.offDiag x) (Membership.mem (H …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


@[simp]
theorem prod_sdiff_diagonal : s ×ˢ s \ diagonal α = s.offDiag :=
  ext fun _ => and_assoc


@[simp]
theorem disjoint_diagonal_offDiag : Disjoint (diagonal α) s.offDiag :=
  disjoint_left.mpr fun _ hd ho => ho.2.2 hd


theorem offDiag_inter : (s ∩ t).offDiag = s.offDiag ∩ t.offDiag :=
  ext fun x => by
    /-
      α : Type u_1
      s t : Set α
      x : Prod α α
      ⊢ Iff (Membership.mem (Inter.inter s t).offDiag x) (Membership.mem (Inter.inte …
    -/
    simp only [mem_offDiag, mem_inter_iff]
    /-
      α : Type u_1
      s t : Set α
      x : Prod α α
      ⊢ Iff (And (And (Membership.mem s x.1) (Membership.mem t x.1)) (And (And (Memb …
    -/
    tauto
    /-
      🎉 no goals
    -/


theorem offDiag_union (h : Disjoint s t) :
    (s ∪ t).offDiag = s.offDiag ∪ t.offDiag ∪ s ×ˢ t ∪ t ×ˢ s := by
  /-
    α : Type u_1
    s t : Set α
    h : Disjoint s t
    ⊢ Eq (Union.union s t).offDiag (Union.union (Union.union (Union.union s.offDia …
  -/
  ext x
  /-
    case h
    α : Type u_1
    s t : Set α
    h : Disjoint s t
    x : Prod α α
    ⊢ Iff (Membership.mem (Union.union s t).offDiag x) (Membership.mem (Union.unio …
  -/
  simp only [mem_offDiag, mem_union, ne_eq, mem_prod]
  /-
    case h
    α : Type u_1
    s t : Set α
    h : Disjoint s t
    x : Prod α α
    ⊢ Iff (And (Or (Membership.mem s x.1) (Membership.mem t x.1)) (And (Or (Member …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      s t : Set α
      h : Disjoint s t
      x : Prod α α
      ⊢ And (Or (Membership.mem s x.1) (Membership.mem t x.1)) (And (Or (Membership. …
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
  · rintro ⟨h0|h0, h1|h1, h2⟩ <;> simp [h0, h1, h2]
                                  /-
                                    🎉 no goals
                                  -/
    /-
      case h.mpr
      α : Type u_1
      s t : Set α
      h : Disjoint s t
      x : Prod α α
      ⊢ Or (Or (Or (And (Membership.mem s x.1) (And (Membership.mem s x.2) (Not (Eq  …
    -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
  · rintro (((⟨h0, h1, h2⟩|⟨h0, h1, h2⟩)|⟨h0, h1⟩)|⟨h0, h1⟩) <;> simp [*]
      /-
        case h.mpr.inl.inr.intro
        α : Type u_1
        s t : Set α
        h : Disjoint s t
        x : Prod α α
        h0 : Membership.mem s x.1
        h1 : Membership.mem t x.2
        ⊢ Not (Eq x.1 x.2)
      -/
    · rintro h3
      /-
        case h.mpr.inl.inr.intro
        α : Type u_1
        s t : Set α
        h : Disjoint s t
        x : Prod α α
        h0 : Membership.mem s x.1
        h1 : Membership.mem t x.2
        h3 : Eq x.1 x.2
        ⊢ False
      -/
      rw [h3] at h0
      /-
        case h.mpr.inl.inr.intro
        α : Type u_1
        s t : Set α
        h : Disjoint s t
        x : Prod α α
        h0 : Membership.mem s x.2
        h1 : Membership.mem t x.2
        h3 : Eq x.1 x.2
        ⊢ False
      -/
      exact Set.disjoint_left.mp h h0 h1
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.inr.intro
        α : Type u_1
        s t : Set α
        h : Disjoint s t
        x : Prod α α
        h0 : Membership.mem t x.1
        h1 : Membership.mem s x.2
        ⊢ Not (Eq x.1 x.2)
      -/
    · rintro h3
      /-
        case h.mpr.inr.intro
        α : Type u_1
        s t : Set α
        h : Disjoint s t
        x : Prod α α
        h0 : Membership.mem t x.1
        h1 : Membership.mem s x.2
        h3 : Eq x.1 x.2
        ⊢ False
      -/
      rw [h3] at h0
      /-
        case h.mpr.inr.intro
        α : Type u_1
        s t : Set α
        h : Disjoint s t
        x : Prod α α
        h0 : Membership.mem t x.2
        h1 : Membership.mem s x.2
        h3 : Eq x.1 x.2
        ⊢ False
      -/
      exact (Set.disjoint_right.mp h h0 h1).elim
      /-
        🎉 no goals
      -/


theorem offDiag_insert (ha : a ∉ s) : (insert a s).offDiag = s.offDiag ∪ {a} ×ˢ s ∪ s ×ˢ {a} := by
  /-
    α : Type u_1
    s : Set α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Eq (Insert.insert a s).offDiag (Union.union (Union.union s.offDiag (SProd.sp …
  -/
  rw [insert_eq, union_comm, offDiag_union, offDiag_singleton, union_empty, union_right_comm]
  /-
    α : Type u_1
    s : Set α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ Disjoint s (Singleton.singleton a)
  -/
  rw [disjoint_left]
  /-
    α : Type u_1
    s : Set α
    a : α
    ha : Not (Membership.mem s a)
    ⊢ ∀ ⦃a_1 : α⦄, Membership.mem s a_1 → Not (Membership.mem (Singleton.singleton …
  -/
  rintro b hb (rfl : b = a)
  /-
    α : Type u_1
    s : Set α
    b : α
    hb : Membership.mem s b
    ha : Not (Membership.mem s b)
    ⊢ False
  -/
  exact ha hb
  /-
    🎉 no goals
  -/


@[simp]
theorem empty_pi (s : ∀ i, Set (α i)) : pi ∅ s = univ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : (i : ι) → Set (α i)
    ⊢ Eq (EmptyCollection.emptyCollection.pi s) Set.univ
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : (i : ι) → Set (α i)
    x✝ : (i : ι) → α i
    ⊢ Iff (Membership.mem (EmptyCollection.emptyCollection.pi s) x✝) (Membership.m …
  -/
  simp [pi]
  /-
    🎉 no goals
  -/


theorem subsingleton_univ_pi (ht : ∀ i, (t i).Subsingleton) :
    (univ.pi t).Subsingleton := fun _f hf _g hg ↦ funext fun i ↦
  (ht i) (hf _ <| mem_univ _) (hg _ <| mem_univ _)


@[simp]
theorem pi_univ (s : Set ι) : (pi s fun i => (univ : Set (α i))) = univ :=
  eq_univ_of_forall fun _ _ _ => mem_univ _


@[simp]
theorem pi_univ_ite (s : Set ι) [DecidablePred (· ∈ s)] (t : ∀ i, Set (α i)) :
    (pi univ fun i => if i ∈ s then t i else univ) = s.pi t := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    inst✝ : DecidablePred fun x => Membership.mem s x
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.univ.pi fun i => ite (Membership.mem s i) (t i) Set.univ) (s.pi t)
  -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  ext; simp_rw [Set.mem_pi]; apply forall_congr'; intro i; split_ifs with h <;> simp [h]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem pi_mono (h : ∀ i ∈ s, t₁ i ⊆ t₂ i) : pi s t₁ ⊆ pi s t₂ := fun _ hx i hi => h i hi <| hx i hi


theorem pi_inter_distrib : (s.pi fun i => t i ∩ t₁ i) = s.pi t ∩ s.pi t₁ :=
                  /-
                    ι : Type u_1
                    α : ι → Type u_2
                    s : Set ι
                    t t₁ : (i : ι) → Set (α i)
                    x : (i : ι) → α i
                    ⊢ Iff (Membership.mem (s.pi fun i => Inter.inter (t i) (t₁ i)) x) (Membership. …
                  -/
  ext fun x => by simp only [forall_and, mem_pi, mem_inter_iff]
                  /-
                    🎉 no goals
                  -/


theorem pi_congr (h : s₁ = s₂) (h' : ∀ i ∈ s₁, t₁ i = t₂ i) : s₁.pi t₁ = s₂.pi t₂ :=
  h ▸ ext fun _ => forall₂_congr fun i hi => h' i hi ▸ Iff.rfl


theorem pi_eq_empty (hs : i ∈ s) (ht : t i = ∅) : s.pi t = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    hs : Membership.mem s i
    ht : Eq (t i) EmptyCollection.emptyCollection
    ⊢ Eq (s.pi t) EmptyCollection.emptyCollection
  -/
  ext f
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    hs : Membership.mem s i
    ht : Eq (t i) EmptyCollection.emptyCollection
    f : (i : ι) → α i
    ⊢ Iff (Membership.mem (s.pi t) f) (Membership.mem EmptyCollection.emptyCollect …
  -/
  simp only [mem_empty_iff_false, not_forall, iff_false, mem_pi, Classical.not_imp]
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    hs : Membership.mem s i
    ht : Eq (t i) EmptyCollection.emptyCollection
    f : (i : ι) → α i
    ⊢ Exists fun x => Exists fun x_1 => Not (Membership.mem (t x) (f x))
  -/
  exact ⟨i, hs, by simp [ht]⟩
  /-
    🎉 no goals
  -/


theorem univ_pi_eq_empty (ht : t i = ∅) : pi univ t = ∅ :=
  pi_eq_empty (mem_univ i) ht


theorem pi_nonempty_iff : (s.pi t).Nonempty ↔ ∀ i, ∃ x, i ∈ s → x ∈ t i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    ⊢ Iff (s.pi t).Nonempty (∀ (i : ι), Exists fun x => Membership.mem s i → Membe …
  -/
  simp [Classical.skolem, Set.Nonempty]
  /-
    🎉 no goals
  -/


theorem univ_pi_nonempty_iff : (pi univ t).Nonempty ↔ ∀ i, (t i).Nonempty := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    t : (i : ι) → Set (α i)
    ⊢ Iff (Set.univ.pi t).Nonempty (∀ (i : ι), (t i).Nonempty)
  -/
  simp [Classical.skolem, Set.Nonempty]
  /-
    🎉 no goals
  -/


theorem pi_eq_empty_iff : s.pi t = ∅ ↔ ∃ i, IsEmpty (α i) ∨ i ∈ s ∧ t i = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    ⊢ Iff (Eq (s.pi t) EmptyCollection.emptyCollection) (Exists fun i => Or (IsEmp …
  -/
  rw [← not_nonempty_iff_eq_empty, pi_nonempty_iff]
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    ⊢ Iff (Not (∀ (i : ι), Exists fun x => Membership.mem s i → Membership.mem (t  …
  -/
  push_neg
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    ⊢ Iff (Exists fun i => ∀ (x : α i), And (Membership.mem s i) (Not (Membership. …
  -/
  refine exists_congr fun i => ?_
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    ⊢ Iff (∀ (x : α i), And (Membership.mem s i) (Not (Membership.mem (t i) x))) ( …
  -/
                                      /-
                                        🎉 no goals
                                      -/
  cases isEmpty_or_nonempty (α i) <;> simp [*, forall_and, eq_empty_iff_forall_not_mem]
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem univ_pi_eq_empty_iff : pi univ t = ∅ ↔ ∃ i, t i = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    t : (i : ι) → Set (α i)
    ⊢ Iff (Eq (Set.univ.pi t) EmptyCollection.emptyCollection) (Exists fun i => Eq …
  -/
  simp [← not_nonempty_iff_eq_empty, univ_pi_nonempty_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem univ_pi_empty [h : Nonempty ι] : pi univ (fun _ => ∅ : ∀ i, Set (α i)) = ∅ :=
  univ_pi_eq_empty_iff.2 <| h.elim fun x => ⟨x, rfl⟩


@[simp]
theorem disjoint_univ_pi : Disjoint (pi univ t₁) (pi univ t₂) ↔ ∃ i, Disjoint (t₁ i) (t₂ i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    t₁ t₂ : (i : ι) → Set (α i)
    ⊢ Iff (Disjoint (Set.univ.pi t₁) (Set.univ.pi t₂)) (Exists fun i => Disjoint ( …
  -/
  simp only [disjoint_iff_inter_eq_empty, ← pi_inter_distrib, univ_pi_eq_empty_iff]
  /-
    🎉 no goals
  -/


theorem Disjoint.set_pi (hi : i ∈ s) (ht : Disjoint (t₁ i) (t₂ i)) : Disjoint (s.pi t₁) (s.pi t₂) :=
  disjoint_left.2 fun _ h₁ h₂ => disjoint_left.1 ht (h₁ _ hi) (h₂ _ hi)


theorem uniqueElim_preimage [Unique ι] (t : ∀ i, Set (α i)) :
                                                     /-
                                                       ι : Type u_1
                                                       α : ι → Type u_2
                                                       inst✝ : Unique ι
                                                       t : (i : ι) → Set (α i)
                                                       ⊢ Eq (Set.preimage uniqueElim (Set.univ.pi t)) (t Inhabited.default)
                                                     -/
    uniqueElim ⁻¹' pi univ t = t (default : ι) := by ext; simp [Unique.forall_iff]
                                                          /-
                                                            🎉 no goals
                                                          -/


                                                               /-
                                                                 ι : Type u_1
                                                                 α : ι → Type u_2
                                                                 s : Set ι
                                                                 t : (i : ι) → Set (α i)
                                                                 inst✝ : ∀ (i : ι), Nonempty (α i)
                                                                 ⊢ Iff (Eq (s.pi t) EmptyCollection.emptyCollection) (Exists fun i => And (Memb …
                                                               -/
theorem pi_eq_empty_iff' : s.pi t = ∅ ↔ ∃ i ∈ s, t i = ∅ := by simp [pi_eq_empty_iff]
                                                               /-
                                                                 🎉 no goals
                                                               -/


@[simp]
theorem disjoint_pi : Disjoint (s.pi t₁) (s.pi t₂) ↔ ∃ i ∈ s, Disjoint (t₁ i) (t₂ i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    inst✝ : ∀ (i : ι), Nonempty (α i)
    ⊢ Iff (Disjoint (s.pi t₁) (s.pi t₂)) (Exists fun i => And (Membership.mem s i) …
  -/
  simp only [disjoint_iff_inter_eq_empty, ← pi_inter_distrib, pi_eq_empty_iff']
  /-
    🎉 no goals
  -/


@[simp]
theorem insert_pi (i : ι) (s : Set ι) (t : ∀ i, Set (α i)) :
    pi (insert i s) t = eval i ⁻¹' t i ∩ pi s t := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    s : Set ι
    t : (i : ι) → Set (α i)
    ⊢ Eq ((Insert.insert i s).pi t) (Inter.inter (Set.preimage (Function.eval i) ( …
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    s : Set ι
    t : (i : ι) → Set (α i)
    x✝ : (i : ι) → α i
    ⊢ Iff (Membership.mem ((Insert.insert i s).pi t) x✝) (Membership.mem (Inter.in …
  -/
  simp [pi, or_imp, forall_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem singleton_pi (i : ι) (t : ∀ i, Set (α i)) : pi {i} t = eval i ⁻¹' t i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    t : (i : ι) → Set (α i)
    ⊢ Eq ((Singleton.singleton i).pi t) (Set.preimage (Function.eval i) (t i))
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    t : (i : ι) → Set (α i)
    x✝ : (i : ι) → α i
    ⊢ Iff (Membership.mem ((Singleton.singleton i).pi t) x✝) (Membership.mem (Set. …
  -/
  simp [pi]
  /-
    🎉 no goals
  -/


theorem singleton_pi' (i : ι) (t : ∀ i, Set (α i)) : pi {i} t = { x | x i ∈ t i } :=
  singleton_pi i t


theorem univ_pi_singleton (f : ∀ i, α i) : (pi univ fun i => {f i}) = ({f} : Set (∀ i, α i)) :=
                  /-
                    ι : Type u_1
                    α : ι → Type u_2
                    f g : (i : ι) → α i
                    ⊢ Iff (Membership.mem (Set.univ.pi fun i => Singleton.singleton (f i)) g) (Mem …
                  -/
  ext fun g => by simp [funext_iff]
                  /-
                    🎉 no goals
                  -/


theorem preimage_pi (s : Set ι) (t : ∀ i, Set (β i)) (f : ∀ i, α i → β i) :
    (fun (g : ∀ i, α i) i => f _ (g i)) ⁻¹' s.pi t = s.pi fun i => f i ⁻¹' t i :=
  rfl


theorem pi_if {p : ι → Prop} [h : DecidablePred p] (s : Set ι) (t₁ t₂ : ∀ i, Set (α i)) :
    (pi s fun i => if p i then t₁ i else t₂ i) =
      pi ({ i ∈ s | p i }) t₁ ∩ pi ({ i ∈ s | ¬p i }) t₂ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    p : ι → Prop
    h : DecidablePred p
    s : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    ⊢ Eq (s.pi fun i => ite (p i) (t₁ i) (t₂ i)) (Inter.inter ((setOf fun i => And …
  -/
  ext f
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    p : ι → Prop
    h : DecidablePred p
    s : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    f : (i : ι) → α i
    ⊢ Iff (Membership.mem (s.pi fun i => ite (p i) (t₁ i) (t₂ i)) f) (Membership.m …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case h.refine_1
      ι : Type u_1
      α : ι → Type u_2
      p : ι → Prop
      h✝ : DecidablePred p
      s : Set ι
      t₁ t₂ : (i : ι) → Set (α i)
      f : (i : ι) → α i
      h : Membership.mem (s.pi fun i => ite (p i) (t₁ i) (t₂ i)) f
      ⊢ Membership.mem (Inter.inter ((setOf fun i => And (Membership.mem s i) (p i)) …
    -/
  · constructor <;>
        /-
          case h.refine_1.left
          ι : Type u_1
          α : ι → Type u_2
          p : ι → Prop
          h✝ : DecidablePred p
          s : Set ι
          t₁ t₂ : (i : ι) → Set (α i)
          f : (i : ι) → α i
          h : Membership.mem (s.pi fun i => ite (p i) (t₁ i) (t₂ i)) f
          ⊢ Membership.mem ((setOf fun i => And (Membership.mem s i) (p i)).pi t₁) f
        -/
        /-
          case h.refine_1.left.intro
          ι : Type u_1
          α : ι → Type u_2
          p : ι → Prop
          h✝ : DecidablePred p
          s : Set ι
          t₁ t₂ : (i : ι) → Set (α i)
          f : (i : ι) → α i
          h : Membership.mem (s.pi fun i => ite (p i) (t₁ i) (t₂ i)) f
          i : ι
          his : Membership.mem s i
          hpi : p i
          ⊢ Membership.mem (t₁ i) (f i)
        -/
        /-
          🎉 no goals
        -/
        /-
          case h.refine_1.right.intro
          ι : Type u_1
          α : ι → Type u_2
          p : ι → Prop
          h✝ : DecidablePred p
          s : Set ι
          t₁ t₂ : (i : ι) → Set (α i)
          f : (i : ι) → α i
          h : Membership.mem (s.pi fun i => ite (p i) (t₁ i) (t₂ i)) f
          i : ι
          his : Membership.mem s i
          hpi : Not (p i)
          ⊢ Membership.mem (t₂ i) (f i)
        -/
        simpa [*] using h i
        /-
          🎉 no goals
        -/
    /-
      case h.refine_2
      ι : Type u_1
      α : ι → Type u_2
      p : ι → Prop
      h : DecidablePred p
      s : Set ι
      t₁ t₂ : (i : ι) → Set (α i)
      f : (i : ι) → α i
      ⊢ Membership.mem (Inter.inter ((setOf fun i => And (Membership.mem s i) (p i)) …
    -/
  · rintro ⟨ht₁, ht₂⟩ i his
    /-
      case h.refine_2.intro
      ι : Type u_1
      α : ι → Type u_2
      p : ι → Prop
      h : DecidablePred p
      s : Set ι
      t₁ t₂ : (i : ι) → Set (α i)
      f : (i : ι) → α i
      ht₁ : Membership.mem ((setOf fun i => And (Membership.mem s i) (p i)).pi t₁) f
      ht₂ : Membership.mem ((setOf fun i => And (Membership.mem s i) (Not (p i))).pi …
      i : ι
      his : Membership.mem s i
      ⊢ Membership.mem ((fun i => ite (p i) (t₁ i) (t₂ i)) i) (f i)
    -/
                     /-
                       🎉 no goals
                     -/
    by_cases p i <;> simp_all
                     /-
                       🎉 no goals
                     -/


theorem union_pi : (s₁ ∪ s₂).pi t = s₁.pi t ∩ s₂.pi t := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s₁ s₂ : Set ι
    t : (i : ι) → Set (α i)
    ⊢ Eq ((Union.union s₁ s₂).pi t) (Inter.inter (s₁.pi t) (s₂.pi t))
  -/
  simp [pi, or_imp, forall_and, setOf_and]
  /-
    🎉 no goals
  -/


theorem union_pi_inter
    (ht₁ : ∀ i ∉ s₁, t₁ i = univ) (ht₂ : ∀ i ∉ s₂, t₂ i = univ) :
    (s₁ ∪ s₂).pi (fun i ↦ t₁ i ∩ t₂ i) = s₁.pi t₁ ∩ s₂.pi t₂ := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s₁ s₂ : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
    ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
    ⊢ Eq ((Union.union s₁ s₂).pi fun i => Inter.inter (t₁ i) (t₂ i)) (Inter.inter  …
  -/
  ext x
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s₁ s₂ : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
    ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
    x : (i : ι) → α i
    ⊢ Iff (Membership.mem ((Union.union s₁ s₂).pi fun i => Inter.inter (t₁ i) (t₂  …
  -/
  simp only [mem_pi, mem_union, mem_inter_iff]
  refine ⟨fun h ↦ ⟨fun i his₁ ↦ (h i (Or.inl his₁)).1, fun i his₂ ↦ (h i (Or.inr his₂)).2⟩,
    fun h i hi ↦ ?_⟩
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s₁ s₂ : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
    ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
    x : (i : ι) → α i
    h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
    i : ι
    hi : Or (Membership.mem s₁ i) (Membership.mem s₂ i)
    ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
  -/
  rcases hi with hi | hi
    /-
      case h.inl
      ι : Type u_1
      α : ι → Type u_2
      s₁ s₂ : Set ι
      t₁ t₂ : (i : ι) → Set (α i)
      ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
      ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
      x : (i : ι) → α i
      h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
      i : ι
      hi : Membership.mem s₁ i
      ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
    -/
  · by_cases hi2 : i ∈ s₂
      /-
        case pos
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₁ i
        hi2 : Membership.mem s₂ i
        ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
      -/
    · exact ⟨h.1 i hi, h.2 i hi2⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₁ i
        hi2 : Not (Membership.mem s₂ i)
        ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
      -/
    · refine ⟨h.1 i hi, ?_⟩
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₁ i
        hi2 : Not (Membership.mem s₂ i)
        ⊢ Membership.mem (t₂ i) (x i)
      -/
      rw [ht₂ i hi2]
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₁ i
        hi2 : Not (Membership.mem s₂ i)
        ⊢ Membership.mem Set.univ (x i)
      -/
      exact mem_univ _
      /-
        🎉 no goals
      -/
    /-
      case h.inr
      ι : Type u_1
      α : ι → Type u_2
      s₁ s₂ : Set ι
      t₁ t₂ : (i : ι) → Set (α i)
      ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
      ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
      x : (i : ι) → α i
      h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
      i : ι
      hi : Membership.mem s₂ i
      ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
    -/
  · by_cases hi1 : i ∈ s₁
      /-
        case pos
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₂ i
        hi1 : Membership.mem s₁ i
        ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
      -/
    · exact ⟨h.1 i hi1, h.2 i hi⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₂ i
        hi1 : Not (Membership.mem s₁ i)
        ⊢ And (Membership.mem (t₁ i) (x i)) (Membership.mem (t₂ i) (x i))
      -/
    · refine ⟨?_, h.2 i hi⟩
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₂ i
        hi1 : Not (Membership.mem s₁ i)
        ⊢ Membership.mem (t₁ i) (x i)
      -/
      rw [ht₁ i hi1]
      /-
        case neg
        ι : Type u_1
        α : ι → Type u_2
        s₁ s₂ : Set ι
        t₁ t₂ : (i : ι) → Set (α i)
        ht₁ : ∀ (i : ι), Not (Membership.mem s₁ i) → Eq (t₁ i) Set.univ
        ht₂ : ∀ (i : ι), Not (Membership.mem s₂ i) → Eq (t₂ i) Set.univ
        x : (i : ι) → α i
        h : And (∀ (i : ι), Membership.mem s₁ i → Membership.mem (t₁ i) (x i)) (∀ (i : …
        i : ι
        hi : Membership.mem s₂ i
        hi1 : Not (Membership.mem s₁ i)
        ⊢ Membership.mem Set.univ (x i)
      -/
      exact mem_univ _
      /-
        🎉 no goals
      -/


@[simp]
theorem pi_inter_compl (s : Set ι) : pi s t ∩ pi sᶜ t = pi univ t := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    t : (i : ι) → Set (α i)
    s : Set ι
    ⊢ Eq (Inter.inter (s.pi t) ((HasCompl.compl s).pi t)) (Set.univ.pi t)
  -/
  rw [← union_pi, union_compl_self]
  /-
    🎉 no goals
  -/


theorem pi_update_of_not_mem [DecidableEq ι] (hi : i ∉ s) (f : ∀ j, α j) (a : α i)
    (t : ∀ j, α j → Set (β j)) : (s.pi fun j => t j (update f i a j)) = s.pi fun j => t j (f j) :=
  (pi_congr rfl) fun j hj => by
    /-
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      s : Set ι
      i : ι
      inst✝ : DecidableEq ι
      hi : Not (Membership.mem s i)
      f : (j : ι) → α j
      a : α i
      t : (j : ι) → α j → Set (β j)
      j : ι
      hj : Membership.mem s j
      ⊢ Eq (t j (Function.update f i a j)) (t j (f j))
    -/
    rw [update_of_ne]
    /-
      case h
      ι : Type u_1
      α : ι → Type u_2
      β : ι → Type u_3
      s : Set ι
      i : ι
      inst✝ : DecidableEq ι
      hi : Not (Membership.mem s i)
      f : (j : ι) → α j
      a : α i
      t : (j : ι) → α j → Set (β j)
      j : ι
      hj : Membership.mem s j
      ⊢ Ne j i
    -/
    exact fun h => hi (h ▸ hj)
    /-
      🎉 no goals
    -/


theorem pi_update_of_mem [DecidableEq ι] (hi : i ∈ s) (f : ∀ j, α j) (a : α i)
    (t : ∀ j, α j → Set (β j)) :
    (s.pi fun j => t j (update f i a j)) = { x | x i ∈ t i a } ∩ (s \ {i}).pi fun j => t j (f j) :=
  calc
    (s.pi fun j => t j (update f i a j)) = ({i} ∪ s \ {i}).pi fun j => t j (update f i a j) := by
        /-
          ι : Type u_1
          α : ι → Type u_2
          β : ι → Type u_3
          s : Set ι
          i : ι
          inst✝ : DecidableEq ι
          hi : Membership.mem s i
          f : (j : ι) → α j
          a : α i
          t : (j : ι) → α j → Set (β j)
          ⊢ Eq (s.pi fun j => t j (Function.update f i a j)) ((Union.union (Singleton.si …
        -/
        rw [union_diff_self, union_eq_self_of_subset_left (singleton_subset_iff.2 hi)]
        /-
          🎉 no goals
        -/
    _ = { x | x i ∈ t i a } ∩ (s \ {i}).pi fun j => t j (f j) := by
        /-
          ι : Type u_1
          α : ι → Type u_2
          β : ι → Type u_3
          s : Set ι
          i : ι
          inst✝ : DecidableEq ι
          hi : Membership.mem s i
          f : (j : ι) → α j
          a : α i
          t : (j : ι) → α j → Set (β j)
          ⊢ Eq ((Union.union (Singleton.singleton i) (SDiff.sdiff s (Singleton.singleton …
        -/
        rw [union_pi, singleton_pi', update_self, pi_update_of_not_mem]; simp
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem univ_pi_update [DecidableEq ι] {β : ι → Type*} (i : ι) (f : ∀ j, α j) (a : α i)
    (t : ∀ j, α j → Set (β j)) :
    (pi univ fun j => t j (update f i a j)) = { x | x i ∈ t i a } ∩ pi {i}ᶜ fun j => t j (f j) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : DecidableEq ι
    β : ι → Type u_4
    i : ι
    f : (j : ι) → α j
    a : α i
    t : (j : ι) → α j → Set (β j)
    ⊢ Eq (Set.univ.pi fun j => t j (Function.update f i a j)) (Inter.inter (setOf  …
  -/
  rw [compl_eq_univ_diff, ← pi_update_of_mem (mem_univ _)]
  /-
    🎉 no goals
  -/


theorem univ_pi_update_univ [DecidableEq ι] (i : ι) (s : Set (α i)) :
    pi univ (update (fun j : ι => (univ : Set (α j))) i s) = eval i ⁻¹' s := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    inst✝ : DecidableEq ι
    i : ι
    s : Set (α i)
    ⊢ Eq (Set.univ.pi (Function.update (fun j => Set.univ) i s)) (Set.preimage (Fu …
  -/
  rw [univ_pi_update i (fun j => (univ : Set (α j))) s fun j t => t, pi_univ, inter_univ, preimage]
  /-
    🎉 no goals
  -/


theorem eval_image_pi_subset (hs : i ∈ s) : eval i '' s.pi t ⊆ t i :=
  image_subset_iff.2 fun _ hf => hf i hs


theorem eval_image_univ_pi_subset : eval i '' pi univ t ⊆ t i :=
  eval_image_pi_subset (mem_univ i)


theorem subset_eval_image_pi (ht : (s.pi t).Nonempty) (i : ι) : t i ⊆ eval i '' s.pi t := by
  classical
  obtain ⟨f, hf⟩ := ht
  refine fun y hy => ⟨update f i y, fun j hj => ?_, update_self ..⟩
  obtain rfl | hji := eq_or_ne j i <;> simp [*, hf _ hj]


theorem eval_image_pi (hs : i ∈ s) (ht : (s.pi t).Nonempty) : eval i '' s.pi t = t i :=
  (eval_image_pi_subset hs).antisymm (subset_eval_image_pi ht i)


lemma eval_image_pi_of_not_mem [Decidable (s.pi t).Nonempty] (hi : i ∉ s) :
    eval i '' s.pi t = if (s.pi t).Nonempty then univ else ∅ := by
  classical
  ext xᵢ
  simp only [eval, mem_image, mem_pi, Set.Nonempty, mem_ite_empty_right, mem_univ, and_true]
  constructor
  · rintro ⟨x, hx, rfl⟩
    exact ⟨x, hx⟩
  · rintro ⟨x, hx⟩
    refine ⟨Function.update x i xᵢ, ?_⟩
    simpa (config := { contextual := true }) [(ne_of_mem_of_not_mem · hi)]


@[simp]
theorem eval_image_univ_pi (ht : (pi univ t).Nonempty) :
    (fun f : ∀ i, α i => f i) '' pi univ t = t i :=
  eval_image_pi (mem_univ i) ht


theorem piMap_image_pi {f : ∀ i, α i → β i} (hf : ∀ i ∉ s, Surjective (f i)) (t : ∀ i, Set (α i)) :
    Pi.map f '' s.pi t = s.pi fun i ↦ f i '' t i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    s : Set ι
    f : (i : ι) → α i → β i
    hf : ∀ (i : ι), Not (Membership.mem s i) → Function.Surjective (f i)
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.image (Pi.map f) (s.pi t)) (s.pi fun i => Set.image (f i) (t i))
  -/
  refine Subset.antisymm (image_subset_iff.2 fun a ha i hi ↦ mem_image_of_mem _ (ha _ hi)) ?_
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    s : Set ι
    f : (i : ι) → α i → β i
    hf : ∀ (i : ι), Not (Membership.mem s i) → Function.Surjective (f i)
    t : (i : ι) → Set (α i)
    ⊢ HasSubset.Subset (s.pi fun i => Set.image (f i) (t i)) (Set.image (Pi.map f) …
  -/
  intro b hb
  have : ∀ i, ∃ a, f i a = b i ∧ (i ∈ s → a ∈ t i) := by
    intro i
    if hi : i ∈ s then
      exact (hb i hi).imp fun a ⟨hat, hab⟩ ↦ ⟨hab, fun _ ↦ hat⟩
    else
      exact (hf i hi (b i)).imp fun a ha ↦ ⟨ha, (absurd · hi)⟩
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    s : Set ι
    f : (i : ι) → α i → β i
    hf : ∀ (i : ι), Not (Membership.mem s i) → Function.Surjective (f i)
    t : (i : ι) → Set (α i)
    b : (i : ι) → β i
    hb : Membership.mem (s.pi fun i => Set.image (f i) (t i)) b
    this : ∀ (i : ι), Exists fun a => And (Eq (f i a) (b i)) (Membership.mem s i → …
    ⊢ Membership.mem (Set.image (Pi.map f) (s.pi t)) b
  -/
  choose a hab hat using this
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    s : Set ι
    f : (i : ι) → α i → β i
    hf : ∀ (i : ι), Not (Membership.mem s i) → Function.Surjective (f i)
    t : (i : ι) → Set (α i)
    b : (i : ι) → β i
    hb : Membership.mem (s.pi fun i => Set.image (f i) (t i)) b
    a : (i : ι) → α i
    hab : ∀ (i : ι), Eq (f i (a i)) (b i)
    hat : ∀ (i : ι), Membership.mem s i → Membership.mem (t i) (a i)
    ⊢ Membership.mem (Set.image (Pi.map f) (s.pi t)) b
  -/
  exact ⟨a, hat, funext hab⟩
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-06")] alias dcomp_image_pi := piMap_image_pi


theorem piMap_image_univ_pi (f : ∀ i, α i → β i) (t : ∀ i, Set (α i)) :
    Pi.map f '' univ.pi t = univ.pi fun i ↦ f i '' t i :=
                     /-
                       ι : Type u_1
                       α : ι → Type u_2
                       β : ι → Type u_3
                       f : (i : ι) → α i → β i
                       t : (i : ι) → Set (α i)
                       ⊢ ∀ (i : ι), Not (Membership.mem Set.univ i) → Function.Surjective (f i)
                     -/
  piMap_image_pi (by simp) t
                     /-
                       🎉 no goals
                     -/


@[deprecated (since := "2024-10-06")] alias dcomp_image_univ_pi := piMap_image_univ_pi


@[simp]
theorem range_piMap (f : ∀ i, α i → β i) : range (Pi.map f) = pi univ fun i ↦ range (f i) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    β : ι → Type u_3
    f : (i : ι) → α i → β i
    ⊢ Eq (Set.range (Pi.map f)) (Set.univ.pi fun i => Set.range (f i))
  -/
  simp only [← image_univ, ← piMap_image_univ_pi, pi_univ]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-06")] alias range_dcomp := range_piMap


theorem pi_subset_pi_iff : pi s t₁ ⊆ pi s t₂ ↔ (∀ i ∈ s, t₁ i ⊆ t₂ i) ∨ pi s t₁ = ∅ := by
  refine
    ⟨fun h => or_iff_not_imp_right.2 ?_, fun h => h.elim pi_mono fun h' => h'.symm ▸ empty_subset _⟩
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    h : HasSubset.Subset (s.pi t₁) (s.pi t₂)
    ⊢ Not (Eq (s.pi t₁) EmptyCollection.emptyCollection) → ∀ (i : ι), Membership.m …
  -/
  rw [← Ne, ← nonempty_iff_ne_empty]
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    h : HasSubset.Subset (s.pi t₁) (s.pi t₂)
    ⊢ (s.pi t₁).Nonempty → ∀ (i : ι), Membership.mem s i → HasSubset.Subset (t₁ i) …
  -/
  intro hne i hi
  simpa only [eval_image_pi hi hne, eval_image_pi hi (hne.mono h)] using
    image_subset (fun f : ∀ i, α i => f i) h


theorem univ_pi_subset_univ_pi_iff :
                                                                       /-
                                                                         ι : Type u_1
                                                                         α : ι → Type u_2
                                                                         t₁ t₂ : (i : ι) → Set (α i)
                                                                         ⊢ Iff (HasSubset.Subset (Set.univ.pi t₁) (Set.univ.pi t₂)) (Or (∀ (i : ι), Has …
                                                                       -/
    pi univ t₁ ⊆ pi univ t₂ ↔ (∀ i, t₁ i ⊆ t₂ i) ∨ ∃ i, t₁ i = ∅ := by simp [pi_subset_pi_iff]
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem eval_preimage [DecidableEq ι] {s : Set (α i)} :
    eval i ⁻¹' s = pi univ (update (fun _ => univ) i s) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    inst✝ : DecidableEq ι
    s : Set (α i)
    ⊢ Eq (Set.preimage (Function.eval i) s) (Set.univ.pi (Function.update (fun x = …
  -/
  ext x
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    inst✝ : DecidableEq ι
    s : Set (α i)
    x : (x : ι) → α x
    ⊢ Iff (Membership.mem (Set.preimage (Function.eval i) s) x) (Membership.mem (S …
  -/
  simp [@forall_update_iff _ (fun i => Set (α i)) _ _ _ _ fun i' y => x i' ∈ y]
  /-
    🎉 no goals
  -/


theorem eval_preimage' [DecidableEq ι] {s : Set (α i)} :
    eval i ⁻¹' s = pi {i} (update (fun _ => univ) i s) := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    inst✝ : DecidableEq ι
    s : Set (α i)
    ⊢ Eq (Set.preimage (Function.eval i) s) ((Singleton.singleton i).pi (Function. …
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    i : ι
    inst✝ : DecidableEq ι
    s : Set (α i)
    x✝ : (x : ι) → α x
    ⊢ Iff (Membership.mem (Set.preimage (Function.eval i) s) x✝) (Membership.mem ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem update_preimage_pi [DecidableEq ι] {f : ∀ i, α i} (hi : i ∈ s)
    (hf : ∀ j ∈ s, j ≠ i → f j ∈ t j) : update f i ⁻¹' s.pi t = t i := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    inst✝ : DecidableEq ι
    f : (i : ι) → α i
    hi : Membership.mem s i
    hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
    ⊢ Eq (Set.preimage (Function.update f i) (s.pi t)) (t i)
  -/
  ext x
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    inst✝ : DecidableEq ι
    f : (i : ι) → α i
    hi : Membership.mem s i
    hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
    x : α i
    ⊢ Iff (Membership.mem (Set.preimage (Function.update f i) (s.pi t)) x) (Member …
  -/
  refine ⟨fun h => ?_, fun hx j hj => ?_⟩
    /-
      case h.refine_1
      ι : Type u_1
      α : ι → Type u_2
      s : Set ι
      t : (i : ι) → Set (α i)
      i : ι
      inst✝ : DecidableEq ι
      f : (i : ι) → α i
      hi : Membership.mem s i
      hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
      x : α i
      h : Membership.mem (Set.preimage (Function.update f i) (s.pi t)) x
      ⊢ Membership.mem (t i) x
    -/
  · convert h i hi
    /-
      case h.e'_5
      ι : Type u_1
      α : ι → Type u_2
      s : Set ι
      t : (i : ι) → Set (α i)
      i : ι
      inst✝ : DecidableEq ι
      f : (i : ι) → α i
      hi : Membership.mem s i
      hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
      x : α i
      h : Membership.mem (Set.preimage (Function.update f i) (s.pi t)) x
      ⊢ Eq x (Function.update f i x i)
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      ι : Type u_1
      α : ι → Type u_2
      s : Set ι
      t : (i : ι) → Set (α i)
      i : ι
      inst✝ : DecidableEq ι
      f : (i : ι) → α i
      hi : Membership.mem s i
      hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
      x : α i
      hx : Membership.mem (t i) x
      j : ι
      hj : Membership.mem s j
      ⊢ Membership.mem (t j) (Function.update f i x j)
    -/
  · obtain rfl | h := eq_or_ne j i
      /-
        case h.refine_2.inl
        ι : Type u_1
        α : ι → Type u_2
        s : Set ι
        t : (i : ι) → Set (α i)
        inst✝ : DecidableEq ι
        f : (i : ι) → α i
        j : ι
        hj hi : Membership.mem s j
        hf : ∀ (j_1 : ι), Membership.mem s j_1 → Ne j_1 j → Membership.mem (t j_1) (f  …
        x : α j
        hx : Membership.mem (t j) x
        ⊢ Membership.mem (t j) (Function.update f j x j)
      -/
    · simpa
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.inr
        ι : Type u_1
        α : ι → Type u_2
        s : Set ι
        t : (i : ι) → Set (α i)
        i : ι
        inst✝ : DecidableEq ι
        f : (i : ι) → α i
        hi : Membership.mem s i
        hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
        x : α i
        hx : Membership.mem (t i) x
        j : ι
        hj : Membership.mem s j
        h : Ne j i
        ⊢ Membership.mem (t j) (Function.update f i x j)
      -/
    · rw [update_of_ne h]
      /-
        case h.refine_2.inr
        ι : Type u_1
        α : ι → Type u_2
        s : Set ι
        t : (i : ι) → Set (α i)
        i : ι
        inst✝ : DecidableEq ι
        f : (i : ι) → α i
        hi : Membership.mem s i
        hf : ∀ (j : ι), Membership.mem s j → Ne j i → Membership.mem (t j) (f j)
        x : α i
        hx : Membership.mem (t i) x
        j : ι
        hj : Membership.mem s j
        h : Ne j i
        ⊢ Membership.mem (t j) (f j)
      -/
      exact hf j hj h
      /-
        🎉 no goals
      -/


theorem update_image [DecidableEq ι] (x : (i : ι) → β i) (i : ι) (s : Set (β i)) :
    update x i '' s = Set.univ.pi (update (fun j ↦ {x j}) i s) := by
  /-
    ι : Type u_1
    β : ι → Type u_3
    inst✝ : DecidableEq ι
    x : (i : ι) → β i
    i : ι
    s : Set (β i)
    ⊢ Eq (Set.image (Function.update x i) s) (Set.univ.pi (Function.update (fun j  …
  -/
  ext y
  simp only [mem_image, update_eq_iff, ne_eq, and_left_comm (a := _ ∈ s), exists_eq_left, mem_pi,
    mem_univ, true_implies]
  /-
    case h
    ι : Type u_1
    β : ι → Type u_3
    inst✝ : DecidableEq ι
    x : (i : ι) → β i
    i : ι
    s : Set (β i)
    y : (a : ι) → β a
    ⊢ Iff (And (Membership.mem s (y i)) (∀ (x_1 : ι), Not (Eq x_1 i) → Eq (x x_1)  …
  -/
  rw [forall_update_iff (p := fun x s => y x ∈ s)]
  /-
    case h
    ι : Type u_1
    β : ι → Type u_3
    inst✝ : DecidableEq ι
    x : (i : ι) → β i
    i : ι
    s : Set (β i)
    y : (a : ι) → β a
    ⊢ Iff (And (Membership.mem s (y i)) (∀ (x_1 : ι), Not (Eq x_1 i) → Eq (x x_1)  …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


theorem update_preimage_univ_pi [DecidableEq ι] {f : ∀ i, α i} (hf : ∀ j ≠ i, f j ∈ t j) :
    update f i ⁻¹' pi univ t = t i :=
  update_preimage_pi (mem_univ i) fun j _ => hf j


theorem subset_pi_eval_image (s : Set ι) (u : Set (∀ i, α i)) : u ⊆ pi s fun i => eval i '' u :=
  fun f hf _ _ => ⟨f, hf, rfl⟩


theorem univ_pi_ite (s : Set ι) [DecidablePred (· ∈ s)] (t : ∀ i, Set (α i)) :
    (pi univ fun i => if i ∈ s then t i else univ) = s.pi t := by
  /-
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    inst✝ : DecidablePred fun x => Membership.mem s x
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.univ.pi fun i => ite (Membership.mem s i) (t i) Set.univ) (s.pi t)
  -/
  ext
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    inst✝ : DecidablePred fun x => Membership.mem s x
    t : (i : ι) → Set (α i)
    x✝ : (i : ι) → α i
    ⊢ Iff (Membership.mem (Set.univ.pi fun i => ite (Membership.mem s i) (t i) Set …
  -/
  simp_rw [mem_univ_pi]
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    inst✝ : DecidablePred fun x => Membership.mem s x
    t : (i : ι) → Set (α i)
    x✝ : (i : ι) → α i
    ⊢ Iff (∀ (i : ι), Membership.mem (ite (Membership.mem s i) (t i) Set.univ) (x✝ …
  -/
  refine forall_congr' fun i => ?_
  /-
    case h
    ι : Type u_1
    α : ι → Type u_2
    s : Set ι
    inst✝ : DecidablePred fun x => Membership.mem s x
    t : (i : ι) → Set (α i)
    x✝ : (i : ι) → α i
    i : ι
    ⊢ Iff (Membership.mem (ite (Membership.mem s i) (t i) Set.univ) (x✝ i)) (Membe …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem piCongrLeft_symm_preimage_pi (f : ι' ≃ ι) (s : Set ι') (t : ∀ i, Set (α i)) :
    (f.piCongrLeft α).symm ⁻¹' s.pi (fun i' => t <| f i') = (f '' s).pi t := by
  /-
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    f : Equiv ι' ι
    s : Set ι'
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.preimage (⇑(Equiv.piCongrLeft α f).symm) (s.pi fun i' => t (f i')))  …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem piCongrLeft_symm_preimage_univ_pi (f : ι' ≃ ι) (t : ∀ i, Set (α i)) :
    (f.piCongrLeft α).symm ⁻¹' univ.pi (fun i' => t <| f i') = univ.pi t := by
  /-
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    f : Equiv ι' ι
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.preimage (⇑(Equiv.piCongrLeft α f).symm) (Set.univ.pi fun i' => t (f …
  -/
  simpa [f.surjective.range_eq] using piCongrLeft_symm_preimage_pi f univ t
  /-
    🎉 no goals
  -/


theorem piCongrLeft_preimage_pi (f : ι' ≃ ι) (s : Set ι') (t : ∀ i, Set (α i)) :
    f.piCongrLeft α ⁻¹' (f '' s).pi t = s.pi fun i => t (f i) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    f : Equiv ι' ι
    s : Set ι'
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.preimage (⇑(Equiv.piCongrLeft α f)) ((Set.image (⇑f) s).pi t)) (s.pi …
  -/
  apply Set.ext
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    f : Equiv ι' ι
    s : Set ι'
    t : (i : ι) → Set (α i)
    ⊢ ∀ (x : (a : ι') → α (f a)), Iff (Membership.mem (Set.preimage (⇑(Equiv.piCon …
  -/
  rw [← (f.piCongrLeft α).symm.forall_congr_right]
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    f : Equiv ι' ι
    s : Set ι'
    t : (i : ι) → Set (α i)
    ⊢ ∀ (a : (b : ι) → α b), Iff (Membership.mem (Set.preimage (⇑(Equiv.piCongrLef …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem piCongrLeft_preimage_univ_pi (f : ι' ≃ ι) (t : ∀ i, Set (α i)) :
    f.piCongrLeft α ⁻¹' univ.pi t = univ.pi fun i => t (f i) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    f : Equiv ι' ι
    t : (i : ι) → Set (α i)
    ⊢ Eq (Set.preimage (⇑(Equiv.piCongrLeft α f)) (Set.univ.pi t)) (Set.univ.pi fu …
  -/
  simpa [f.surjective.range_eq] using piCongrLeft_preimage_pi f univ t
  /-
    🎉 no goals
  -/


theorem sumPiEquivProdPi_symm_preimage_univ_pi (π : ι ⊕ ι' → Type*) (t : ∀ i, Set (π i)) :
    (sumPiEquivProdPi π).symm ⁻¹' univ.pi t =
    univ.pi (fun i => t (.inl i)) ×ˢ univ.pi fun i => t (.inr i) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    π : Sum ι ι' → Type u_4
    t : (i : Sum ι ι') → Set (π i)
    ⊢ Eq (Set.preimage (⇑(Equiv.sumPiEquivProdPi π).symm) (Set.univ.pi t)) (SProd. …
  -/
  ext
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    π : Sum ι ι' → Type u_4
    t : (i : Sum ι ι') → Set (π i)
    x✝ : Prod ((i : ι) → π (Sum.inl i)) ((i' : ι') → π (Sum.inr i'))
    ⊢ Iff (Membership.mem (Set.preimage (⇑(Equiv.sumPiEquivProdPi π).symm) (Set.un …
  -/
  simp_rw [mem_preimage, mem_prod, mem_univ_pi, sumPiEquivProdPi_symm_apply]
  /-
    case h
    ι : Type u_1
    ι' : Type u_2
    π : Sum ι ι' → Type u_4
    t : (i : Sum ι ι') → Set (π i)
    x✝ : Prod ((i : ι) → π (Sum.inl i)) ((i' : ι') → π (Sum.inr i'))
    ⊢ Iff (∀ (i : Sum ι ι'), Membership.mem (t i) (Sum.rec x✝.1 x✝.2 i)) (And (∀ ( …
  -/
  constructor
    /-
      case h.mp
      ι : Type u_1
      ι' : Type u_2
      π : Sum ι ι' → Type u_4
      t : (i : Sum ι ι') → Set (π i)
      x✝ : Prod ((i : ι) → π (Sum.inl i)) ((i' : ι') → π (Sum.inr i'))
      ⊢ (∀ (i : Sum ι ι'), Membership.mem (t i) (Sum.rec x✝.1 x✝.2 i)) → And (∀ (i : …
    -/
                                         /-
                                           🎉 no goals
                                         -/
  · intro h; constructor <;> intro i <;> apply h
                                         /-
                                           🎉 no goals
                                         -/
    /-
      case h.mpr
      ι : Type u_1
      ι' : Type u_2
      π : Sum ι ι' → Type u_4
      t : (i : Sum ι ι') → Set (π i)
      x✝ : Prod ((i : ι) → π (Sum.inl i)) ((i' : ι') → π (Sum.inr i'))
      ⊢ And (∀ (i : ι), Membership.mem (t (Sum.inl i)) (x✝.1 i)) (∀ (i : ι'), Member …
    -/
                                       /-
                                         🎉 no goals
                                       -/
  · rintro ⟨h₁, h₂⟩ (i|i) <;> simp <;> apply_assumption
                                       /-
                                         🎉 no goals
                                       -/


