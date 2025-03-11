@[simp]
theorem range_sigmaMk (i : ι) : range (Sigma.mk i : α i → Sigma α) = Sigma.fst ⁻¹' {i} := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    i : ι
    ⊢ Eq (Set.range (Sigma.mk i)) (Set.preimage Sigma.fst (Singleton.singleton i))
  -/
  apply Subset.antisymm
    /-
      case h₁
      ι : Type u_1
      α : ι → Type u_3
      i : ι
      ⊢ HasSubset.Subset (Set.range (Sigma.mk i)) (Set.preimage Sigma.fst (Singleton …
    -/
  · rintro _ ⟨b, rfl⟩
    /-
      case h₁.intro
      ι : Type u_1
      α : ι → Type u_3
      i : ι
      b : α i
      ⊢ Membership.mem (Set.preimage Sigma.fst (Singleton.singleton i)) ⟨i, b⟩
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case h₂
      ι : Type u_1
      α : ι → Type u_3
      i : ι
      ⊢ HasSubset.Subset (Set.preimage Sigma.fst (Singleton.singleton i)) (Set.range …
    -/
  · rintro ⟨x, y⟩ (rfl | _)
    /-
      case h₂.mk.refl
      ι : Type u_1
      α : ι → Type u_3
      x : ι
      y : α x
      ⊢ Membership.mem (Set.range (Sigma.mk ⟨x, y⟩.fst)) ⟨x, y⟩
    -/
    exact mem_range_self y
    /-
      🎉 no goals
    -/


theorem preimage_image_sigmaMk_of_ne (h : i ≠ j) (s : Set (α j)) :
    Sigma.mk i ⁻¹' (Sigma.mk j '' s) = ∅ := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    i j : ι
    h : Ne i j
    s : Set (α j)
    ⊢ Eq (Set.preimage (Sigma.mk i) (Set.image (Sigma.mk j) s)) EmptyCollection.em …
  -/
  ext x
  /-
    case h
    ι : Type u_1
    α : ι → Type u_3
    i j : ι
    h : Ne i j
    s : Set (α j)
    x : α i
    ⊢ Iff (Membership.mem (Set.preimage (Sigma.mk i) (Set.image (Sigma.mk j) s)) x …
  -/
  simp [h.symm]
  /-
    🎉 no goals
  -/


theorem image_sigmaMk_preimage_sigmaMap_subset {β : ι' → Type*} (f : ι → ι')
    (g : ∀ i, α i → β (f i)) (i : ι) (s : Set (β (f i))) :
    Sigma.mk i '' (g i ⁻¹' s) ⊆ Sigma.map f g ⁻¹' (Sigma.mk (f i) '' s) :=
  image_subset_iff.2 fun x hx ↦ ⟨g i x, hx, rfl⟩


theorem image_sigmaMk_preimage_sigmaMap {β : ι' → Type*} {f : ι → ι'} (hf : Function.Injective f)
    (g : ∀ i, α i → β (f i)) (i : ι) (s : Set (β (f i))) :
    Sigma.mk i '' (g i ⁻¹' s) = Sigma.map f g ⁻¹' (Sigma.mk (f i) '' s) := by
  /-
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    β : ι' → Type u_4
    f : ι → ι'
    hf : Function.Injective f
    g : (i : ι) → α i → β (f i)
    i : ι
    s : Set (β (f i))
    ⊢ Eq (Set.image (Sigma.mk i) (Set.preimage (g i) s)) (Set.preimage (Sigma.map  …
  -/
  refine (image_sigmaMk_preimage_sigmaMap_subset f g i s).antisymm ?_
  /-
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    β : ι' → Type u_4
    f : ι → ι'
    hf : Function.Injective f
    g : (i : ι) → α i → β (f i)
    i : ι
    s : Set (β (f i))
    ⊢ HasSubset.Subset (Set.preimage (Sigma.map f g) (Set.image (Sigma.mk (f i)) s …
  -/
  rintro ⟨j, x⟩ ⟨y, hys, hxy⟩
  /-
    case mk.intro.intro
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    β : ι' → Type u_4
    f : ι → ι'
    hf : Function.Injective f
    g : (i : ι) → α i → β (f i)
    i : ι
    s : Set (β (f i))
    j : ι
    x : α j
    y : β (f i)
    hys : Membership.mem s y
    hxy : Eq ⟨f i, y⟩ (Sigma.map f g ⟨j, x⟩)
    ⊢ Membership.mem (Set.image (Sigma.mk i) (Set.preimage (g i) s)) ⟨j, x⟩
  -/
  simp only [hf.eq_iff, Sigma.map, Sigma.ext_iff] at hxy
  /-
    case mk.intro.intro
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    β : ι' → Type u_4
    f : ι → ι'
    hf : Function.Injective f
    g : (i : ι) → α i → β (f i)
    i : ι
    s : Set (β (f i))
    j : ι
    x : α j
    y : β (f i)
    hys : Membership.mem s y
    hxy : And (Eq i j) (HEq y (g j x))
    ⊢ Membership.mem (Set.image (Sigma.mk i) (Set.preimage (g i) s)) ⟨j, x⟩
  -/
  rcases hxy with ⟨rfl, hxy⟩; rw [heq_iff_eq] at hxy; subst y
  /-
    case mk.intro.intro.intro
    ι : Type u_1
    ι' : Type u_2
    α : ι → Type u_3
    β : ι' → Type u_4
    f : ι → ι'
    hf : Function.Injective f
    g : (i : ι) → α i → β (f i)
    i : ι
    s : Set (β (f i))
    x : α i
    hys : Membership.mem s (g i x)
    ⊢ Membership.mem (Set.image (Sigma.mk i) (Set.preimage (g i) s)) ⟨i, x⟩
  -/
  exact ⟨x, hys, rfl⟩
  /-
    🎉 no goals
  -/


/-- Indexed sum of sets. `s.sigma t` is the set of dependent pairs `⟨i, a⟩` such that `i ∈ s` and
`a ∈ t i`. -/
protected def sigma (s : Set ι) (t : ∀ i, Set (α i)) : Set (Σ i, α i) := {x | x.1 ∈ s ∧ x.2 ∈ t x.1}


@[simp] theorem mem_sigma_iff : x ∈ s.sigma t ↔ x.1 ∈ s ∧ x.2 ∈ t x.1 := Iff.rfl


theorem mk_sigma_iff : (⟨i, a⟩ : Σ i, α i) ∈ s.sigma t ↔ i ∈ s ∧ a ∈ t i := Iff.rfl


theorem mk_mem_sigma (hi : i ∈ s) (ha : a ∈ t i) : (⟨i, a⟩ : Σ i, α i) ∈ s.sigma t := ⟨hi, ha⟩


theorem sigma_mono (hs : s₁ ⊆ s₂) (ht : ∀ i, t₁ i ⊆ t₂ i) : s₁.sigma t₁ ⊆ s₂.sigma t₂ := fun _ hx ↦
  ⟨hs hx.1, ht _ hx.2⟩


theorem sigma_subset_iff :
    s.sigma t ⊆ u ↔ ∀ ⦃i⦄, i ∈ s → ∀ ⦃a⦄, a ∈ t i → (⟨i, a⟩ : Σ i, α i) ∈ u :=
  ⟨fun h _ hi _ ha ↦ h <| mk_mem_sigma hi ha, fun h _ ha ↦ h ha.1 ha.2⟩


theorem forall_sigma_iff {p : (Σ i, α i) → Prop} :
    (∀ x ∈ s.sigma t, p x) ↔ ∀ ⦃i⦄, i ∈ s → ∀ ⦃a⦄, a ∈ t i → p ⟨i, a⟩ := sigma_subset_iff


theorem exists_sigma_iff {p : (Σi, α i) → Prop} :
    (∃ x ∈ s.sigma t, p x) ↔ ∃ i ∈ s, ∃ a ∈ t i, p ⟨i, a⟩ :=
  ⟨fun ⟨⟨i, a⟩, ha, h⟩ ↦ ⟨i, ha.1, a, ha.2, h⟩, fun ⟨i, hi, a, ha, h⟩ ↦ ⟨⟨i, a⟩, ⟨hi, ha⟩, h⟩⟩


@[simp] theorem sigma_empty : s.sigma (fun i ↦ (∅ : Set (α i))) = ∅ :=
  ext fun _ ↦ iff_of_eq (and_false _)


@[simp] theorem empty_sigma : (∅ : Set ι).sigma t = ∅ := ext fun _ ↦ iff_of_eq (false_and _)


theorem univ_sigma_univ : (@univ ι).sigma (fun _ ↦ @univ (α i)) = univ :=
  ext fun _ ↦ iff_of_eq (true_and _)


@[simp]
theorem sigma_univ : s.sigma (fun _ ↦ univ : ∀ i, Set (α i)) = Sigma.fst ⁻¹' s :=
  ext fun _ ↦ iff_of_eq (and_true _)


@[simp] theorem univ_sigma_preimage_mk (s : Set (Σ i, α i)) :
    (univ : Set ι).sigma (fun i ↦ Sigma.mk i ⁻¹' s) = s :=
            /-
              ι : Type u_1
              α : ι → Type u_3
              s : Set (Sigma fun i => α i)
              ⊢ ∀ (x : Sigma fun i => α i), Iff (Membership.mem (Set.univ.sigma fun i => Set …
            -/
  ext <| by simp
            /-
              🎉 no goals
            -/


@[simp]
theorem singleton_sigma : ({i} : Set ι).sigma t = Sigma.mk i '' t i :=
  ext fun x ↦ by
    /-
      ι : Type u_1
      α : ι → Type u_3
      t : (i : ι) → Set (α i)
      i : ι
      x : Sigma fun i => α i
      ⊢ Iff (Membership.mem ((Singleton.singleton i).sigma t) x) (Membership.mem (Se …
    -/
    constructor
      /-
        case mp
        ι : Type u_1
        α : ι → Type u_3
        t : (i : ι) → Set (α i)
        i : ι
        x : Sigma fun i => α i
        ⊢ Membership.mem ((Singleton.singleton i).sigma t) x → Membership.mem (Set.ima …
      -/
    · obtain ⟨j, a⟩ := x
      /-
        case mp.mk
        ι : Type u_1
        α : ι → Type u_3
        t : (i : ι) → Set (α i)
        i j : ι
        a : α j
        ⊢ Membership.mem ((Singleton.singleton i).sigma t) ⟨j, a⟩ → Membership.mem (Se …
      -/
      rintro ⟨rfl : j = i, ha⟩
      /-
        case mp.mk.intro
        ι : Type u_1
        α : ι → Type u_3
        t : (i : ι) → Set (α i)
        j : ι
        a : α j
        ha : Membership.mem (t ⟨j, a⟩.fst) ⟨j, a⟩.snd
        ⊢ Membership.mem (Set.image (Sigma.mk j) (t j)) ⟨j, a⟩
      -/
      exact mem_image_of_mem _ ha
      /-
        🎉 no goals
      -/
      /-
        case mpr
        ι : Type u_1
        α : ι → Type u_3
        t : (i : ι) → Set (α i)
        i : ι
        x : Sigma fun i => α i
        ⊢ Membership.mem (Set.image (Sigma.mk i) (t i)) x → Membership.mem ((Singleton …
      -/
    · rintro ⟨b, hb, rfl⟩
      /-
        case mpr.intro.intro
        ι : Type u_1
        α : ι → Type u_3
        t : (i : ι) → Set (α i)
        i : ι
        b : α i
        hb : Membership.mem (t i) b
        ⊢ Membership.mem ((Singleton.singleton i).sigma t) ⟨i, b⟩
      -/
      exact ⟨rfl, hb⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem sigma_singleton {a : ∀ i, α i} :
    s.sigma (fun i ↦ ({a i} : Set (α i))) = (fun i ↦ Sigma.mk i <| a i) '' s := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    s : Set ι
    a : (i : ι) → α i
    ⊢ Eq (s.sigma fun i => Singleton.singleton (a i)) (Set.image (fun i => ⟨i, a i …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    ι : Type u_1
    α : ι → Type u_3
    s : Set ι
    a : (i : ι) → α i
    x : ι
    y : α x
    ⊢ Iff (Membership.mem (s.sigma fun i => Singleton.singleton (a i)) ⟨x, y⟩) (Me …
  -/
  simp [and_left_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem singleton_sigma_singleton {a : ∀ i, α i} :
    (({i} : Set ι).sigma fun i ↦ ({a i} : Set (α i))) = {⟨i, a i⟩} := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    i : ι
    a : (i : ι) → α i
    ⊢ Eq ((Singleton.singleton i).sigma fun i => Singleton.singleton (a i)) (Singl …
  -/
  rw [sigma_singleton, image_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem union_sigma : (s₁ ∪ s₂).sigma t = s₁.sigma t ∪ s₂.sigma t := ext fun _ ↦ or_and_right


@[simp]
theorem sigma_union : s.sigma (fun i ↦ t₁ i ∪ t₂ i) = s.sigma t₁ ∪ s.sigma t₂ :=
  ext fun _ ↦ and_or_left


theorem sigma_inter_sigma : s₁.sigma t₁ ∩ s₂.sigma t₂ = (s₁ ∩ s₂).sigma fun i ↦ t₁ i ∩ t₂ i := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    s₁ s₂ : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    ⊢ Eq (Inter.inter (s₁.sigma t₁) (s₂.sigma t₂)) ((Inter.inter s₁ s₂).sigma fun  …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    ι : Type u_1
    α : ι → Type u_3
    s₁ s₂ : Set ι
    t₁ t₂ : (i : ι) → Set (α i)
    x : ι
    y : α x
    ⊢ Iff (Membership.mem (Inter.inter (s₁.sigma t₁) (s₂.sigma t₂)) ⟨x, y⟩) (Membe …
  -/
  simp [and_assoc, and_left_comm]
  /-
    🎉 no goals
  -/


theorem _root_.biSup_sigma (s : Set ι) (t : ∀ i, Set (α i)) (f : Sigma α → β) :
    ⨆ ij ∈ s.sigma t, f ij = ⨆ (i ∈ s) (j ∈ t i), f ⟨i, j⟩ :=
                                  /-
                                    ι : Type u_1
                                    α : ι → Type u_3
                                    β : Type u_4
                                    inst✝ : CompleteLattice β
                                    s : Set ι
                                    t : (i : ι) → Set (α i)
                                    f : Sigma α → β
                                    x✝ : β
                                    ⊢ LE.le (iSup fun ij => iSup fun h => f ij) x✝ → LE.le (iSup fun i => iSup fun …
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  eq_of_forall_ge_iff fun _ ↦ ⟨by simp_all, by simp_all⟩
                                               /-
                                                 🎉 no goals
                                               -/


theorem _root_.biSup_sigma' (s : Set ι) (t : ∀ i, Set (α i)) (f : ∀ i, α i → β) :
    ⨆ (i ∈ s) (j ∈ t i), f i j = ⨆ ij ∈ s.sigma t, f ij.fst ij.snd :=
  Eq.symm (biSup_sigma _ _ _)


theorem _root_.biInf_sigma (s : Set ι) (t : ∀ i, Set (α i)) (f : Sigma α → β) :
    ⨅ ij ∈ s.sigma t, f ij = ⨅ (i ∈ s) (j ∈ t i), f ⟨i, j⟩ :=
  biSup_sigma (β := βᵒᵈ) _ _ _


theorem _root_.biInf_sigma' (s : Set ι) (t : ∀ i, Set (α i)) (f : ∀ i, α i → β) :
    ⨅ (i ∈ s) (j ∈ t i), f i j = ⨅ ij ∈ s.sigma t, f ij.fst ij.snd :=
  Eq.symm (biInf_sigma _ _ _)


theorem biUnion_sigma (s : Set ι) (t : ∀ i, Set (α i)) (f : Sigma α → Set β) :
    ⋃ ij ∈ s.sigma t, f ij = ⋃ i ∈ s, ⋃ j ∈ t i, f ⟨i, j⟩ :=
  biSup_sigma _ _ _


theorem biUnion_sigma' (s : Set ι) (t : ∀ i, Set (α i)) (f : ∀ i, α i → Set β) :
    ⋃ i ∈ s, ⋃ j ∈ t i, f i j = ⋃ ij ∈ s.sigma t, f ij.fst ij.snd :=
  biSup_sigma' _ _ _


theorem biInter_sigma (s : Set ι) (t : ∀ i, Set (α i)) (f : Sigma α → Set β) :
    ⋂ ij ∈ s.sigma t, f ij = ⋂ i ∈ s, ⋂ j ∈ t i, f ⟨i, j⟩ :=
  biInf_sigma _ _ _


theorem biInter_sigma' (s : Set ι) (t : ∀ i, Set (α i)) (f : ∀ i, α i → Set β) :
    ⋂ i ∈ s, ⋂ j ∈ t i, f i j = ⋂ ij ∈ s.sigma t, f ij.fst ij.snd :=
  biInf_sigma' _ _ _


theorem insert_sigma : (insert i s).sigma t = Sigma.mk i '' t i ∪ s.sigma t := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    s : Set ι
    t : (i : ι) → Set (α i)
    i : ι
    ⊢ Eq ((Insert.insert i s).sigma t) (Union.union (Set.image (Sigma.mk i) (t i)) …
  -/
  rw [insert_eq, union_sigma, singleton_sigma]
  /-
    🎉 no goals
  -/


theorem sigma_insert {a : ∀ i, α i} :
    s.sigma (fun i ↦ insert (a i) (t i)) = (fun i ↦ ⟨i, a i⟩) '' s ∪ s.sigma t := by
  /-
    ι : Type u_1
    α : ι → Type u_3
    s : Set ι
    t : (i : ι) → Set (α i)
    a : (i : ι) → α i
    ⊢ Eq (s.sigma fun i => Insert.insert (a i) (t i)) (Union.union (Set.image (fun …
  -/
  simp_rw [insert_eq, sigma_union, sigma_singleton]
  /-
    🎉 no goals
  -/


theorem sigma_preimage_eq {f : ι' → ι} {g : ∀ i, β i → α i} :
    (f ⁻¹' s).sigma (fun i ↦ g (f i) ⁻¹' t (f i)) =
      (fun p : Σ i, β (f i) ↦ Sigma.mk _ (g _ p.2)) ⁻¹' s.sigma t := rfl


theorem sigma_preimage_left {f : ι' → ι} :
    ((f ⁻¹' s).sigma fun i ↦ t (f i)) = (fun p : Σ i, α (f i) ↦ Sigma.mk _ p.2) ⁻¹' s.sigma t :=
  rfl


theorem sigma_preimage_right {g : ∀ i, β i → α i} :
    (s.sigma fun i ↦ g i ⁻¹' t i) = (fun p : Σ i, β i ↦ Sigma.mk p.1 (g _ p.2)) ⁻¹' s.sigma t :=
  rfl


theorem preimage_sigmaMap_sigma {α' : ι' → Type*} (f : ι → ι') (g : ∀ i, α i → α' (f i))
    (s : Set ι') (t : ∀ i, Set (α' i)) :
    Sigma.map f g ⁻¹' s.sigma t = (f ⁻¹' s).sigma fun i ↦ g i ⁻¹' t (f i) := rfl


@[simp]
theorem mk_preimage_sigma (hi : i ∈ s) : Sigma.mk i ⁻¹' s.sigma t = t i :=
  ext fun _ ↦ and_iff_right hi


@[simp]
theorem mk_preimage_sigma_eq_empty (hi : i ∉ s) : Sigma.mk i ⁻¹' s.sigma t = ∅ :=
  ext fun _ ↦ iff_of_false (hi ∘ And.left) id


theorem mk_preimage_sigma_eq_if [DecidablePred (· ∈ s)] :
                                                              /-
                                                                ι : Type u_1
                                                                α : ι → Type u_3
                                                                s : Set ι
                                                                t : (i : ι) → Set (α i)
                                                                i : ι
                                                                inst✝ : DecidablePred fun x => Membership.mem s x
                                                                ⊢ Eq (Set.preimage (Sigma.mk i) (s.sigma t)) (ite (Membership.mem s i) (t i) E …
                                                              -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    Sigma.mk i ⁻¹' s.sigma t = if i ∈ s then t i else ∅ := by split_ifs <;> simp [*]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem mk_preimage_sigma_fn_eq_if {β : Type*} [DecidablePred (· ∈ s)] (g : β → α i) :
    (fun b ↦ Sigma.mk i (g b)) ⁻¹' s.sigma t = if i ∈ s then g ⁻¹' t i else ∅ :=
                 /-
                   ι : Type u_1
                   α : ι → Type u_3
                   s : Set ι
                   t : (i : ι) → Set (α i)
                   i : ι
                   β : Type u_7
                   inst✝ : DecidablePred fun x => Membership.mem s x
                   g : β → α i
                   x✝ : β
                   ⊢ Iff (Membership.mem (Set.preimage (fun b => ⟨i, g b⟩) (s.sigma t)) x✝) (Memb …
                 -/
                               /-
                                 🎉 no goals
                               -/
  ext fun _ ↦ by split_ifs <;> simp [*]
                               /-
                                 🎉 no goals
                               -/


theorem sigma_univ_range_eq {f : ∀ i, α i → β i} :
    (univ : Set ι).sigma (fun i ↦ range (f i)) = range fun x : Σ i, α i ↦ ⟨x.1, f _ x.2⟩ :=
            /-
              ι : Type u_1
              α : ι → Type u_3
              β : ι → Type u_6
              f : (i : ι) → α i → β i
              ⊢ ∀ (x : Sigma fun i => β i), Iff (Membership.mem (Set.univ.sigma fun i => Set …
            -/
  ext <| by simp [range, Sigma.forall]
            /-
              🎉 no goals
            -/


protected theorem Nonempty.sigma :
    s.Nonempty → (∀ i, (t i).Nonempty) → (s.sigma t).Nonempty := fun ⟨i, hi⟩ h ↦
  let ⟨a, ha⟩ := h i
  ⟨⟨i, a⟩, hi, ha⟩


theorem Nonempty.sigma_fst : (s.sigma t).Nonempty → s.Nonempty := fun ⟨x, hx⟩ ↦ ⟨x.1, hx.1⟩


theorem Nonempty.sigma_snd : (s.sigma t).Nonempty → ∃ i ∈ s, (t i).Nonempty :=
  fun ⟨x, hx⟩ ↦ ⟨x.1, hx.1, x.2, hx.2⟩


theorem sigma_nonempty_iff : (s.sigma t).Nonempty ↔ ∃ i ∈ s, (t i).Nonempty :=
  ⟨Nonempty.sigma_snd, fun ⟨i, hi, a, ha⟩ ↦ ⟨⟨i, a⟩, hi, ha⟩⟩


theorem sigma_eq_empty_iff : s.sigma t = ∅ ↔ ∀ i ∈ s, t i = ∅ :=
  not_nonempty_iff_eq_empty.symm.trans <|
    sigma_nonempty_iff.not.trans <| by
      /-
        ι : Type u_1
        α : ι → Type u_3
        s : Set ι
        t : (i : ι) → Set (α i)
        ⊢ Iff (Not (Exists fun i => And (Membership.mem s i) (t i).Nonempty)) (∀ (i :  …
      -/
      simp only [not_nonempty_iff_eq_empty, not_and, not_exists]
      /-
        🎉 no goals
      -/


theorem image_sigmaMk_subset_sigma_left {a : ∀ i, α i} (ha : ∀ i, a i ∈ t i) :
    (fun i ↦ Sigma.mk i (a i)) '' s ⊆ s.sigma t :=
  image_subset_iff.2 fun _ hi ↦ ⟨hi, ha _⟩


theorem image_sigmaMk_subset_sigma_right (hi : i ∈ s) : Sigma.mk i '' t i ⊆ s.sigma t :=
  image_subset_iff.2 fun _ ↦ And.intro hi


theorem sigma_subset_preimage_fst (s : Set ι) (t : ∀ i, Set (α i)) : s.sigma t ⊆ Sigma.fst ⁻¹' s :=
  fun _ ↦ And.left


theorem fst_image_sigma_subset (s : Set ι) (t : ∀ i, Set (α i)) : Sigma.fst '' s.sigma t ⊆ s :=
  image_subset_iff.2 fun _ ↦ And.left


theorem fst_image_sigma (s : Set ι) (ht : ∀ i, (t i).Nonempty) : Sigma.fst '' s.sigma t = s :=
  (fst_image_sigma_subset _ _).antisymm fun i hi ↦
    let ⟨a, ha⟩ := ht i
    ⟨⟨i, a⟩, ⟨hi, ha⟩, rfl⟩


theorem sigma_diff_sigma : s₁.sigma t₁ \ s₂.sigma t₂ = s₁.sigma (t₁ \ t₂) ∪ (s₁ \ s₂).sigma t₁ :=
  ext fun x ↦ by
    /-
      ι : Type u_1
      α : ι → Type u_3
      s₁ s₂ : Set ι
      t₁ t₂ : (i : ι) → Set (α i)
      x : Sigma fun i => α i
      ⊢ Iff (Membership.mem (SDiff.sdiff (s₁.sigma t₁) (s₂.sigma t₂)) x) (Membership …
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
    by_cases h₁ : x.1 ∈ s₁ <;> by_cases h₂ : x.2 ∈ t₁ x.1 <;> simp [*, ← imp_iff_or_not]
                                                              /-
                                                                🎉 no goals
                                                              -/


