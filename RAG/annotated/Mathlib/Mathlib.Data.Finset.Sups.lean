/-- `s ⊻ t` is the finset of elements of the form `a ⊔ b` where `a ∈ s`, `b ∈ t`. -/
protected def hasSups : HasSups (Finset α) :=
  ⟨image₂ (· ⊔ ·)⟩


@[simp]
                                                                 /-
                                                                   α : Type u_2
                                                                   inst✝¹ : DecidableEq α
                                                                   inst✝ : SemilatticeSup α
                                                                   s t : Finset α
                                                                   c : α
                                                                   ⊢ Iff (Membership.mem (HasSups.sups s t) c) (Exists fun a => And (Membership.m …
                                                                 -/
theorem mem_sups : c ∈ s ⊻ t ↔ ∃ a ∈ s, ∃ b ∈ t, a ⊔ b = c := by simp [(· ⊻ ·)]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp, norm_cast]
theorem coe_sups : (↑(s ⊻ t) : Set α) = ↑s ⊻ ↑t :=
  coe_image₂ _ _ _


theorem card_sups_le : #(s ⊻ t) ≤ #s * #t := card_image₂_le _ _ _


theorem card_sups_iff : #(s ⊻ t) = #s * #t ↔ (s ×ˢ t : Set (α × α)).InjOn fun x => x.1 ⊔ x.2 :=
  card_image₂_iff


theorem sup_mem_sups : a ∈ s → b ∈ t → a ⊔ b ∈ s ⊻ t :=
  mem_image₂_of_mem


theorem sups_subset : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ ⊻ t₁ ⊆ s₂ ⊻ t₂ :=
  image₂_subset


theorem sups_subset_left : t₁ ⊆ t₂ → s ⊻ t₁ ⊆ s ⊻ t₂ :=
  image₂_subset_left


theorem sups_subset_right : s₁ ⊆ s₂ → s₁ ⊻ t ⊆ s₂ ⊻ t :=
  image₂_subset_right


lemma image_subset_sups_left : b ∈ t → s.image (· ⊔ b) ⊆ s ⊻ t := image_subset_image₂_left


lemma image_subset_sups_right : a ∈ s → t.image (a ⊔ ·) ⊆ s ⊻ t := image_subset_image₂_right


theorem forall_sups_iff {p : α → Prop} : (∀ c ∈ s ⊻ t, p c) ↔ ∀ a ∈ s, ∀ b ∈ t, p (a ⊔ b) :=
  forall_mem_image₂


@[simp]
theorem sups_subset_iff : s ⊻ t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, a ⊔ b ∈ u :=
  image₂_subset_iff


@[simp]
theorem sups_nonempty : (s ⊻ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  image₂_nonempty_iff


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected theorem Nonempty.sups : s.Nonempty → t.Nonempty → (s ⊻ t).Nonempty :=
  Nonempty.image₂


theorem Nonempty.of_sups_left : (s ⊻ t).Nonempty → s.Nonempty :=
  Nonempty.of_image₂_left


theorem Nonempty.of_sups_right : (s ⊻ t).Nonempty → t.Nonempty :=
  Nonempty.of_image₂_right


@[simp]
theorem empty_sups : ∅ ⊻ t = ∅ :=
  image₂_empty_left


@[simp]
theorem sups_empty : s ⊻ ∅ = ∅ :=
  image₂_empty_right


@[simp]
theorem sups_eq_empty : s ⊻ t = ∅ ↔ s = ∅ ∨ t = ∅ :=
  image₂_eq_empty_iff


@[simp] lemma singleton_sups : {a} ⊻ t = t.image (a ⊔ ·) := image₂_singleton_left


@[simp] lemma sups_singleton : s ⊻ {b} = s.image (· ⊔ b) := image₂_singleton_right


theorem singleton_sups_singleton : ({a} ⊻ {b} : Finset α) = {a ⊔ b} :=
  image₂_singleton


theorem sups_union_left : (s₁ ∪ s₂) ⊻ t = s₁ ⊻ t ∪ s₂ ⊻ t :=
  image₂_union_left


theorem sups_union_right : s ⊻ (t₁ ∪ t₂) = s ⊻ t₁ ∪ s ⊻ t₂ :=
  image₂_union_right


theorem sups_inter_subset_left : (s₁ ∩ s₂) ⊻ t ⊆ s₁ ⊻ t ∩ s₂ ⊻ t :=
  image₂_inter_subset_left


theorem sups_inter_subset_right : s ⊻ (t₁ ∩ t₂) ⊆ s ⊻ t₁ ∩ s ⊻ t₂ :=
  image₂_inter_subset_right


theorem subset_sups {s t : Set α} :
    ↑u ⊆ s ⊻ t → ∃ s' t' : Finset α, ↑s' ⊆ s ∧ ↑t' ⊆ t ∧ u ⊆ s' ⊻ t' :=
  subset_set_image₂


lemma image_sups (f : F) (s t : Finset α) : image f (s ⊻ t) = image f s ⊻ image f t :=
  image_image₂_distrib <| map_sup f


lemma map_sups (f : F) (hf) (s t : Finset α) :
    map ⟨f, hf⟩ (s ⊻ t) = map ⟨f, hf⟩ s ⊻ map ⟨f, hf⟩ t := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁵ : DecidableEq α
    inst✝⁴ : DecidableEq β
    inst✝³ : SemilatticeSup α
    inst✝² : SemilatticeSup β
    inst✝¹ : FunLike F α β
    inst✝ : SupHomClass F α β
    f : F
    hf : Function.Injective ⇑f
    s t : Finset α
    ⊢ Eq (Finset.map { toFun := ⇑f, inj' := hf } (HasSups.sups s t)) (HasSups.sups …
  -/
  simpa [map_eq_image] using image_sups f s t
  /-
    🎉 no goals
  -/


lemma subset_sups_self : s ⊆ s ⊻ s := fun _a ha ↦ mem_sups.2 ⟨_, ha, _, ha, sup_idem _⟩

lemma sups_subset_self : s ⊻ s ⊆ s ↔ SupClosed (s : Set α) := sups_subset_iff

                                                                     /-
                                                                       α : Type u_2
                                                                       inst✝¹ : DecidableEq α
                                                                       inst✝ : SemilatticeSup α
                                                                       s : Finset α
                                                                       ⊢ Iff (Eq (HasSups.sups s s) s) (SupClosed ↑s)
                                                                     -/
@[simp] lemma sups_eq_self : s ⊻ s = s ↔ SupClosed (s : Set α) := by simp [← coe_inj]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


                                                                                 /-
                                                                                   α : Type u_2
                                                                                   inst✝² : DecidableEq α
                                                                                   inst✝¹ : SemilatticeSup α
                                                                                   inst✝ : Fintype α
                                                                                   ⊢ Eq (HasSups.sups Finset.univ Finset.univ) Finset.univ
                                                                                 -/
@[simp] lemma univ_sups_univ [Fintype α] : (univ : Finset α) ⊻ univ = univ := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


lemma filter_sups_le [DecidableRel (α := α) (· ≤ ·)] (s t : Finset α) (a : α) :
    {b ∈ s ⊻ t | b ≤ a} = {b ∈ s | b ≤ a} ⊻ {b ∈ t | b ≤ a} := by
  /-
    α : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : SemilatticeSup α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    s t : Finset α
    a : α
    ⊢ Eq (Finset.filter (fun b => LE.le b a) (HasSups.sups s t)) (HasSups.sups (Fi …
  -/
  simp only [← coe_inj, coe_filter, coe_sups, ← mem_coe, Set.sep_sups_le]
  /-
    🎉 no goals
  -/


lemma biUnion_image_sup_left : s.biUnion (fun a ↦ t.image (a ⊔ ·)) = s ⊻ t := biUnion_image_left


lemma biUnion_image_sup_right : t.biUnion (fun b ↦ s.image (· ⊔ b)) = s ⊻ t := biUnion_image_right

-- Porting note: simpNF linter doesn't like @[simp]

theorem image_sup_product (s t : Finset α) : (s ×ˢ t).image (uncurry (· ⊔ ·)) = s ⊻ t :=
  image_uncurry_product _ _ _


theorem sups_assoc : s ⊻ t ⊻ u = s ⊻ (t ⊻ u) := image₂_assoc sup_assoc


theorem sups_comm : s ⊻ t = t ⊻ s := image₂_comm sup_comm


theorem sups_left_comm : s ⊻ (t ⊻ u) = t ⊻ (s ⊻ u) :=
  image₂_left_comm sup_left_comm


theorem sups_right_comm : s ⊻ t ⊻ u = s ⊻ u ⊻ t :=
  image₂_right_comm sup_right_comm


theorem sups_sups_sups_comm : s ⊻ t ⊻ (u ⊻ v) = s ⊻ u ⊻ (t ⊻ v) :=
  image₂_image₂_image₂_comm sup_sup_sup_comm


/-- `s ⊼ t` is the finset of elements of the form `a ⊓ b` where `a ∈ s`, `b ∈ t`. -/
protected def hasInfs : HasInfs (Finset α) :=
  ⟨image₂ (· ⊓ ·)⟩


@[simp]
                                                                 /-
                                                                   α : Type u_2
                                                                   inst✝¹ : DecidableEq α
                                                                   inst✝ : SemilatticeInf α
                                                                   s t : Finset α
                                                                   c : α
                                                                   ⊢ Iff (Membership.mem (HasInfs.infs s t) c) (Exists fun a => And (Membership.m …
                                                                 -/
theorem mem_infs : c ∈ s ⊼ t ↔ ∃ a ∈ s, ∃ b ∈ t, a ⊓ b = c := by simp [(· ⊼ ·)]
                                                                 /-
                                                                   🎉 no goals
                                                                 -/


@[simp, norm_cast]
theorem coe_infs : (↑(s ⊼ t) : Set α) = ↑s ⊼ ↑t :=
  coe_image₂ _ _ _


theorem card_infs_le : #(s ⊼ t) ≤ #s * #t := card_image₂_le _ _ _


theorem card_infs_iff : #(s ⊼ t) = #s * #t ↔ (s ×ˢ t : Set (α × α)).InjOn fun x => x.1 ⊓ x.2 :=
  card_image₂_iff


theorem inf_mem_infs : a ∈ s → b ∈ t → a ⊓ b ∈ s ⊼ t :=
  mem_image₂_of_mem


theorem infs_subset : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ ⊼ t₁ ⊆ s₂ ⊼ t₂ :=
  image₂_subset


theorem infs_subset_left : t₁ ⊆ t₂ → s ⊼ t₁ ⊆ s ⊼ t₂ :=
  image₂_subset_left


theorem infs_subset_right : s₁ ⊆ s₂ → s₁ ⊼ t ⊆ s₂ ⊼ t :=
  image₂_subset_right


lemma image_subset_infs_left : b ∈ t → s.image (· ⊓ b) ⊆ s ⊼ t := image_subset_image₂_left


lemma image_subset_infs_right : a ∈ s → t.image (a ⊓ ·) ⊆ s ⊼ t := image_subset_image₂_right


theorem forall_infs_iff {p : α → Prop} : (∀ c ∈ s ⊼ t, p c) ↔ ∀ a ∈ s, ∀ b ∈ t, p (a ⊓ b) :=
  forall_mem_image₂


@[simp]
theorem infs_subset_iff : s ⊼ t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, a ⊓ b ∈ u :=
  image₂_subset_iff


@[simp]
theorem infs_nonempty : (s ⊼ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  image₂_nonempty_iff


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected theorem Nonempty.infs : s.Nonempty → t.Nonempty → (s ⊼ t).Nonempty :=
  Nonempty.image₂


theorem Nonempty.of_infs_left : (s ⊼ t).Nonempty → s.Nonempty :=
  Nonempty.of_image₂_left


theorem Nonempty.of_infs_right : (s ⊼ t).Nonempty → t.Nonempty :=
  Nonempty.of_image₂_right


@[simp]
theorem empty_infs : ∅ ⊼ t = ∅ :=
  image₂_empty_left


@[simp]
theorem infs_empty : s ⊼ ∅ = ∅ :=
  image₂_empty_right


@[simp]
theorem infs_eq_empty : s ⊼ t = ∅ ↔ s = ∅ ∨ t = ∅ :=
  image₂_eq_empty_iff


@[simp] lemma singleton_infs : {a} ⊼ t = t.image (a ⊓ ·) := image₂_singleton_left


@[simp] lemma infs_singleton : s ⊼ {b} = s.image (· ⊓ b) := image₂_singleton_right


theorem singleton_infs_singleton : ({a} ⊼ {b} : Finset α) = {a ⊓ b} :=
  image₂_singleton


theorem infs_union_left : (s₁ ∪ s₂) ⊼ t = s₁ ⊼ t ∪ s₂ ⊼ t :=
  image₂_union_left


theorem infs_union_right : s ⊼ (t₁ ∪ t₂) = s ⊼ t₁ ∪ s ⊼ t₂ :=
  image₂_union_right


theorem infs_inter_subset_left : (s₁ ∩ s₂) ⊼ t ⊆ s₁ ⊼ t ∩ s₂ ⊼ t :=
  image₂_inter_subset_left


theorem infs_inter_subset_right : s ⊼ (t₁ ∩ t₂) ⊆ s ⊼ t₁ ∩ s ⊼ t₂ :=
  image₂_inter_subset_right


theorem subset_infs {s t : Set α} :
    ↑u ⊆ s ⊼ t → ∃ s' t' : Finset α, ↑s' ⊆ s ∧ ↑t' ⊆ t ∧ u ⊆ s' ⊼ t' :=
  subset_set_image₂


lemma image_infs (f : F) (s t : Finset α) : image f (s ⊼ t) = image f s ⊼ image f t :=
  image_image₂_distrib <| map_inf f


lemma map_infs (f : F) (hf) (s t : Finset α) :
    map ⟨f, hf⟩ (s ⊼ t) = map ⟨f, hf⟩ s ⊼ map ⟨f, hf⟩ t := by
  /-
    F : Type u_1
    α : Type u_2
    β : Type u_3
    inst✝⁵ : DecidableEq α
    inst✝⁴ : DecidableEq β
    inst✝³ : SemilatticeInf α
    inst✝² : SemilatticeInf β
    inst✝¹ : FunLike F α β
    inst✝ : InfHomClass F α β
    f : F
    hf : Function.Injective ⇑f
    s t : Finset α
    ⊢ Eq (Finset.map { toFun := ⇑f, inj' := hf } (HasInfs.infs s t)) (HasInfs.infs …
  -/
  simpa [map_eq_image] using image_infs f s t
  /-
    🎉 no goals
  -/


lemma subset_infs_self : s ⊆ s ⊼ s := fun _a ha ↦ mem_infs.2 ⟨_, ha, _, ha, inf_idem _⟩

lemma infs_self_subset : s ⊼ s ⊆ s ↔ InfClosed (s : Set α) := infs_subset_iff

                                                                  /-
                                                                    α : Type u_2
                                                                    inst✝¹ : DecidableEq α
                                                                    inst✝ : SemilatticeInf α
                                                                    s : Finset α
                                                                    ⊢ Iff (Eq (HasInfs.infs s s) s) (InfClosed ↑s)
                                                                  -/
@[simp] lemma infs_self : s ⊼ s = s ↔ InfClosed (s : Set α) := by simp [← coe_inj]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                                                 /-
                                                                                   α : Type u_2
                                                                                   inst✝² : DecidableEq α
                                                                                   inst✝¹ : SemilatticeInf α
                                                                                   inst✝ : Fintype α
                                                                                   ⊢ Eq (HasInfs.infs Finset.univ Finset.univ) Finset.univ
                                                                                 -/
@[simp] lemma univ_infs_univ [Fintype α] : (univ : Finset α) ⊼ univ = univ := by simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


lemma filter_infs_le [DecidableRel (α := α) (· ≤ ·)] (s t : Finset α) (a : α) :
    {b ∈ s ⊼ t | a ≤ b} = {b ∈ s | a ≤ b} ⊼ {b ∈ t | a ≤ b} := by
  /-
    α : Type u_2
    inst✝² : DecidableEq α
    inst✝¹ : SemilatticeInf α
    inst✝ : DecidableRel fun x1 x2 => LE.le x1 x2
    s t : Finset α
    a : α
    ⊢ Eq (Finset.filter (fun b => LE.le a b) (HasInfs.infs s t)) (HasInfs.infs (Fi …
  -/
  simp only [← coe_inj, coe_filter, coe_infs, ← mem_coe, Set.sep_infs_le]
  /-
    🎉 no goals
  -/


lemma biUnion_image_inf_left : s.biUnion (fun a ↦ t.image (a ⊓ ·)) = s ⊼ t := biUnion_image_left


lemma biUnion_image_inf_right : t.biUnion (fun b ↦ s.image (· ⊓ b)) = s ⊼ t := biUnion_image_right

-- Porting note: simpNF linter doesn't like @[simp]

theorem image_inf_product (s t : Finset α) : (s ×ˢ t).image (uncurry (· ⊓ ·)) = s ⊼ t :=
  image_uncurry_product _ _ _


theorem infs_assoc : s ⊼ t ⊼ u = s ⊼ (t ⊼ u) := image₂_assoc inf_assoc


theorem infs_comm : s ⊼ t = t ⊼ s := image₂_comm inf_comm


theorem infs_left_comm : s ⊼ (t ⊼ u) = t ⊼ (s ⊼ u) :=
  image₂_left_comm inf_left_comm


theorem infs_right_comm : s ⊼ t ⊼ u = s ⊼ u ⊼ t :=
  image₂_right_comm inf_right_comm


theorem infs_infs_infs_comm : s ⊼ t ⊼ (u ⊼ v) = s ⊼ u ⊼ (t ⊼ v) :=
  image₂_image₂_image₂_comm inf_inf_inf_comm


theorem sups_infs_subset_left : s ⊻ t ⊼ u ⊆ (s ⊻ t) ⊼ (s ⊻ u) :=
  image₂_distrib_subset_left sup_inf_left


theorem sups_infs_subset_right : t ⊼ u ⊻ s ⊆ (t ⊻ s) ⊼ (u ⊻ s) :=
  image₂_distrib_subset_right sup_inf_right


theorem infs_sups_subset_left : s ⊼ (t ⊻ u) ⊆ s ⊼ t ⊻ s ⊼ u :=
  image₂_distrib_subset_left inf_sup_left


theorem infs_sups_subset_right : (t ⊻ u) ⊼ s ⊆ t ⊼ s ⊻ u ⊼ s :=
  image₂_distrib_subset_right inf_sup_right


@[simp] lemma powerset_union (s t : Finset α) : (s ∪ t).powerset = s.powerset ⊻ t.powerset := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Union.union s t).powerset (HasSups.sups s.powerset t.powerset)
  -/
  ext u
  /-
    case h
    α : Type u_2
    inst✝ : DecidableEq α
    s t u : Finset α
    ⊢ Iff (Membership.mem (Union.union s t).powerset u) (Membership.mem (HasSups.s …
  -/
  simp only [mem_sups, mem_powerset, le_eq_subset, sup_eq_union]
  /-
    case h
    α : Type u_2
    inst✝ : DecidableEq α
    s t u : Finset α
    ⊢ Iff (HasSubset.Subset u (Union.union s t)) (Exists fun a => And (HasSubset.S …
  -/
  refine ⟨fun h ↦ ⟨_, inter_subset_left (s₂ := u), _, inter_subset_left (s₂ := u), ?_⟩, ?_⟩
    /-
      case h.refine_1
      α : Type u_2
      inst✝ : DecidableEq α
      s t u : Finset α
      h : HasSubset.Subset u (Union.union s t)
      ⊢ Eq (Union.union (Inter.inter s u) (Inter.inter t u)) u
    -/
  · rwa [← union_inter_distrib_right, inter_eq_right]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_2
      inst✝ : DecidableEq α
      s t u : Finset α
      ⊢ (Exists fun a => And (HasSubset.Subset a s) (Exists fun b => And (HasSubset. …
    -/
  · rintro ⟨v, hv, w, hw, rfl⟩
    /-
      case h.refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      s t v : Finset α
      hv : HasSubset.Subset v s
      w : Finset α
      hw : HasSubset.Subset w t
      ⊢ HasSubset.Subset (Union.union v w) (Union.union s t)
    -/
    exact union_subset_union hv hw
    /-
      🎉 no goals
    -/


@[simp] lemma powerset_inter (s t : Finset α) : (s ∩ t).powerset = s.powerset ⊼ t.powerset := by
  /-
    α : Type u_2
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Inter.inter s t).powerset (HasInfs.infs s.powerset t.powerset)
  -/
  ext u
  /-
    case h
    α : Type u_2
    inst✝ : DecidableEq α
    s t u : Finset α
    ⊢ Iff (Membership.mem (Inter.inter s t).powerset u) (Membership.mem (HasInfs.i …
  -/
  simp only [mem_infs, mem_powerset, le_eq_subset, inf_eq_inter]
  /-
    case h
    α : Type u_2
    inst✝ : DecidableEq α
    s t u : Finset α
    ⊢ Iff (HasSubset.Subset u (Inter.inter s t)) (Exists fun a => And (HasSubset.S …
  -/
  refine ⟨fun h ↦ ⟨_, inter_subset_left (s₂ := u), _, inter_subset_left (s₂ := u), ?_⟩, ?_⟩
    /-
      case h.refine_1
      α : Type u_2
      inst✝ : DecidableEq α
      s t u : Finset α
      h : HasSubset.Subset u (Inter.inter s t)
      ⊢ Eq (Inter.inter (Inter.inter s u) (Inter.inter t u)) u
    -/
  · rwa [← inter_inter_distrib_right, inter_eq_right]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_2
      inst✝ : DecidableEq α
      s t u : Finset α
      ⊢ (Exists fun a => And (HasSubset.Subset a s) (Exists fun b => And (HasSubset. …
    -/
  · rintro ⟨v, hv, w, hw, rfl⟩
    /-
      case h.refine_2.intro.intro.intro.intro
      α : Type u_2
      inst✝ : DecidableEq α
      s t v : Finset α
      hv : HasSubset.Subset v s
      w : Finset α
      hw : HasSubset.Subset w t
      ⊢ HasSubset.Subset (Inter.inter v w) (Inter.inter s t)
    -/
    exact inter_subset_inter hv hw
    /-
      🎉 no goals
    -/


@[simp] lemma powerset_sups_powerset_self (s : Finset α) :
                                               /-
                                                 α : Type u_2
                                                 inst✝ : DecidableEq α
                                                 s : Finset α
                                                 ⊢ Eq (HasSups.sups s.powerset s.powerset) s.powerset
                                               -/
    s.powerset ⊻ s.powerset = s.powerset := by simp [← powerset_union]
                                               /-
                                                 🎉 no goals
                                               -/


@[simp] lemma powerset_infs_powerset_self (s : Finset α) :
                                               /-
                                                 α : Type u_2
                                                 inst✝ : DecidableEq α
                                                 s : Finset α
                                                 ⊢ Eq (HasInfs.infs s.powerset s.powerset) s.powerset
                                               -/
    s.powerset ⊼ s.powerset = s.powerset := by simp [← powerset_inter]
                                               /-
                                                 🎉 no goals
                                               -/


lemma union_mem_sups : s ∈ 𝒜 → t ∈ ℬ → s ∪ t ∈ 𝒜 ⊻ ℬ := sup_mem_sups

lemma inter_mem_infs : s ∈ 𝒜 → t ∈ ℬ → s ∩ t ∈ 𝒜 ⊼ ℬ := inf_mem_infs


/-- The finset of elements of the form `a ⊔ b` where `a ∈ s`, `b ∈ t` and `a` and `b` are disjoint.
-/
def disjSups : Finset α := {ab ∈ s ×ˢ t | Disjoint ab.1 ab.2}.image fun ab => ab.1 ⊔ ab.2


@[inherit_doc]
scoped[FinsetFamily] infixl:74 " ○ " => Finset.disjSups


@[simp]
theorem mem_disjSups : c ∈ s ○ t ↔ ∃ a ∈ s, ∃ b ∈ t, Disjoint a b ∧ a ⊔ b = c := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    c : α
    ⊢ Iff (Membership.mem (s.disjSups t) c) (Exists fun a => And (Membership.mem s …
  -/
  simp [disjSups, and_assoc]
  /-
    🎉 no goals
  -/


theorem disjSups_subset_sups : s ○ t ⊆ s ⊻ t := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ HasSubset.Subset (s.disjSups t) (HasSups.sups s t)
  -/
  simp_rw [subset_iff, mem_sups, mem_disjSups]
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ ∀ ⦃x : α⦄, (Exists fun a => And (Membership.mem s a) (Exists fun b => And (M …
  -/
  exact fun c ⟨a, b, ha, hb, _, hc⟩ => ⟨a, b, ha, hb, hc⟩
  /-
    🎉 no goals
  -/


theorem card_disjSups_le : #(s ○ t) ≤ #s * #t :=
  (card_le_card disjSups_subset_sups).trans <| card_sups_le _ _


theorem disjSups_subset (hs : s₁ ⊆ s₂) (ht : t₁ ⊆ t₂) : s₁ ○ t₁ ⊆ s₂ ○ t₂ :=
  image_subset_image <| filter_subset_filter _ <| product_subset_product hs ht


theorem disjSups_subset_left (ht : t₁ ⊆ t₂) : s ○ t₁ ⊆ s ○ t₂ :=
  disjSups_subset Subset.rfl ht


theorem disjSups_subset_right (hs : s₁ ⊆ s₂) : s₁ ○ t ⊆ s₂ ○ t :=
  disjSups_subset hs Subset.rfl


theorem forall_disjSups_iff {p : α → Prop} :
    (∀ c ∈ s ○ t, p c) ↔ ∀ a ∈ s, ∀ b ∈ t, Disjoint a b → p (a ⊔ b) := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    p : α → Prop
    ⊢ Iff (∀ (c : α), Membership.mem (s.disjSups t) c → p c) (∀ (a : α), Membershi …
  -/
  simp_rw [mem_disjSups]
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    p : α → Prop
    ⊢ Iff (∀ (c : α), (Exists fun a => And (Membership.mem s a) (Exists fun b => A …
  -/
  refine ⟨fun h a ha b hb hab => h _ ⟨_, ha, _, hb, hab, rfl⟩, ?_⟩
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    p : α → Prop
    ⊢ (∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → Disjoint a  …
  -/
  rintro h _ ⟨a, ha, b, hb, hab, rfl⟩
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    p : α → Prop
    h : ∀ (a : α), Membership.mem s a → ∀ (b : α), Membership.mem t b → Disjoint a …
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem t b
    hab : Disjoint a b
    ⊢ p (Max.max a b)
  -/
  exact h _ ha _ hb hab
  /-
    🎉 no goals
  -/


@[simp]
theorem disjSups_subset_iff : s ○ t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, Disjoint a b → a ⊔ b ∈ u :=
  forall_disjSups_iff


theorem Nonempty.of_disjSups_left : (s ○ t).Nonempty → s.Nonempty := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ (s.disjSups t).Nonempty → s.Nonempty
  -/
  simp_rw [Finset.Nonempty, mem_disjSups]
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ (Exists fun x => Exists fun a => And (Membership.mem s a) (Exists fun b => A …
  -/
  exact fun ⟨_, a, ha, _⟩ => ⟨a, ha⟩
  /-
    🎉 no goals
  -/


theorem Nonempty.of_disjSups_right : (s ○ t).Nonempty → t.Nonempty := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ (s.disjSups t).Nonempty → t.Nonempty
  -/
  simp_rw [Finset.Nonempty, mem_disjSups]
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ (Exists fun x => Exists fun a => And (Membership.mem s a) (Exists fun b => A …
  -/
  exact fun ⟨_, _, _, b, hb, _⟩ => ⟨b, hb⟩
  /-
    🎉 no goals
  -/


@[simp]
                                              /-
                                                α : Type u_2
                                                inst✝³ : DecidableEq α
                                                inst✝² : SemilatticeSup α
                                                inst✝¹ : OrderBot α
                                                inst✝ : DecidableRel Disjoint
                                                t : Finset α
                                                ⊢ Eq (EmptyCollection.emptyCollection.disjSups t) EmptyCollection.emptyCollect …
                                              -/
theorem disjSups_empty_left : ∅ ○ t = ∅ := by simp [disjSups]
                                              /-
                                                🎉 no goals
                                              -/


@[simp]
                                               /-
                                                 α : Type u_2
                                                 inst✝³ : DecidableEq α
                                                 inst✝² : SemilatticeSup α
                                                 inst✝¹ : OrderBot α
                                                 inst✝ : DecidableRel Disjoint
                                                 s : Finset α
                                                 ⊢ Eq (s.disjSups EmptyCollection.emptyCollection) EmptyCollection.emptyCollect …
                                               -/
theorem disjSups_empty_right : s ○ ∅ = ∅ := by simp [disjSups]
                                               /-
                                                 🎉 no goals
                                               -/


theorem disjSups_singleton : ({a} ○ {b} : Finset α) = if Disjoint a b then {a ⊔ b} else ∅ := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    a b : α
    ⊢ Eq ((Singleton.singleton a).disjSups (Singleton.singleton b)) (ite (Disjoint …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [disjSups, filter_singleton, h]
                       /-
                         🎉 no goals
                       -/


theorem disjSups_union_left : (s₁ ∪ s₂) ○ t = s₁ ○ t ∪ s₂ ○ t := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s₁ s₂ t : Finset α
    ⊢ Eq ((Union.union s₁ s₂).disjSups t) (Union.union (s₁.disjSups t) (s₂.disjSup …
  -/
  simp [disjSups, filter_union, image_union]
  /-
    🎉 no goals
  -/


theorem disjSups_union_right : s ○ (t₁ ∪ t₂) = s ○ t₁ ∪ s ○ t₂ := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t₁ t₂ : Finset α
    ⊢ Eq (s.disjSups (Union.union t₁ t₂)) (Union.union (s.disjSups t₁) (s.disjSups …
  -/
  simp [disjSups, filter_union, image_union]
  /-
    🎉 no goals
  -/


theorem disjSups_inter_subset_left : (s₁ ∩ s₂) ○ t ⊆ s₁ ○ t ∩ s₂ ○ t := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s₁ s₂ t : Finset α
    ⊢ HasSubset.Subset ((Inter.inter s₁ s₂).disjSups t) (Inter.inter (s₁.disjSups  …
  -/
  simpa only [disjSups, inter_product, filter_inter_distrib] using image_inter_subset _ _ _
  /-
    🎉 no goals
  -/


theorem disjSups_inter_subset_right : s ○ (t₁ ∩ t₂) ⊆ s ○ t₁ ∩ s ○ t₂ := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t₁ t₂ : Finset α
    ⊢ HasSubset.Subset (s.disjSups (Inter.inter t₁ t₂)) (Inter.inter (s.disjSups t …
  -/
  simpa only [disjSups, product_inter, filter_inter_distrib] using image_inter_subset _ _ _
  /-
    🎉 no goals
  -/


theorem disjSups_comm : s ○ t = t ○ s := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    ⊢ Eq (s.disjSups t) (t.disjSups s)
  -/
  ext
  /-
    case h
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem (s.disjSups t) a✝) (Membership.mem (t.disjSups s) a✝)
  -/
  rw [mem_disjSups, mem_disjSups]
  -- Porting note: `exists₂_comm` no longer works with `∃ _ ∈ _, ∃ _ ∈ _, _`
  /-
    case h
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : SemilatticeSup α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t : Finset α
    a✝ : α
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Exists fun b => And (Membersh …
  -/
  constructor <;>
    /-
      case h.mp
      α : Type u_2
      inst✝³ : DecidableEq α
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableRel Disjoint
      s t : Finset α
      a✝ : α
      ⊢ (Exists fun a => And (Membership.mem s a) (Exists fun b => And (Membership.m …
    -/
    /-
      case h.mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝³ : DecidableEq α
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableRel Disjoint
      s t : Finset α
      a✝ a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem t b
      hd : Disjoint a b
      hs : Eq (Max.max a b) a✝
      ⊢ Exists fun a => And (Membership.mem t a) (Exists fun b => And (Membership.me …
    -/
    /-
      case h.mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝³ : DecidableEq α
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableRel Disjoint
      s t : Finset α
      a✝ a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem t b
      hd : Disjoint b a
      hs : Eq (Max.max a b) a✝
      ⊢ Exists fun a => And (Membership.mem t a) (Exists fun b => And (Membership.me …
    -/
    /-
      case h.mp.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝³ : DecidableEq α
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableRel Disjoint
      s t : Finset α
      a✝ a : α
      ha : Membership.mem s a
      b : α
      hb : Membership.mem t b
      hd : Disjoint b a
      hs : Eq (Max.max b a) a✝
      ⊢ Exists fun a => And (Membership.mem t a) (Exists fun b => And (Membership.me …
    -/
    /-
      🎉 no goals
    -/
    /-
      case h.mpr.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝³ : DecidableEq α
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableRel Disjoint
      s t : Finset α
      a✝ a : α
      ha : Membership.mem t a
      b : α
      hb : Membership.mem s b
      hd : Disjoint b a
      hs : Eq (Max.max a b) a✝
      ⊢ Exists fun a => And (Membership.mem s a) (Exists fun b => And (Membership.me …
    -/
    rw [sup_comm] at hs
    /-
      case h.mpr.intro.intro.intro.intro.intro
      α : Type u_2
      inst✝³ : DecidableEq α
      inst✝² : SemilatticeSup α
      inst✝¹ : OrderBot α
      inst✝ : DecidableRel Disjoint
      s t : Finset α
      a✝ a : α
      ha : Membership.mem t a
      b : α
      hb : Membership.mem s b
      hd : Disjoint b a
      hs : Eq (Max.max b a) a✝
      ⊢ Exists fun a => And (Membership.mem s a) (Exists fun b => And (Membership.me …
    -/
    exact ⟨b, hb, a, ha, hd, hs⟩
    /-
      🎉 no goals
    -/


instance : @Std.Commutative (Finset α) (· ○ ·) := ⟨disjSups_comm⟩


theorem disjSups_assoc : ∀ s t u : Finset α, s ○ t ○ u = s ○ (t ○ u) := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    ⊢ ∀ (s t u : Finset α), Eq ((s.disjSups t).disjSups u) (s.disjSups (t.disjSups …
  -/
  refine (associative_of_commutative_of_le inferInstance ?_).assoc
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    ⊢ ∀ (a b c : Finset α), LE.le ((a.disjSups b).disjSups c) (a.disjSups (b.disjS …
  -/
  simp only [le_eq_subset, disjSups_subset_iff, mem_disjSups]
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    ⊢ ∀ (a b c : Finset α) (a_1 : α), (Exists fun a_2 => And (Membership.mem a a_2 …
  -/
  rintro s t u _ ⟨a, ha, b, hb, hab, rfl⟩ c hc habc
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t u : Finset α
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem t b
    hab : Disjoint a b
    c : α
    hc : Membership.mem u c
    habc : Disjoint (Max.max a b) c
    ⊢ Exists fun a_1 => And (Membership.mem s a_1) (Exists fun b_1 => And (Exists  …
  -/
  rw [disjoint_sup_left] at habc
  /-
    case intro.intro.intro.intro.intro
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t u : Finset α
    a : α
    ha : Membership.mem s a
    b : α
    hb : Membership.mem t b
    hab : Disjoint a b
    c : α
    hc : Membership.mem u c
    habc : And (Disjoint a c) (Disjoint b c)
    ⊢ Exists fun a_1 => And (Membership.mem s a_1) (Exists fun b_1 => And (Exists  …
  -/
  exact ⟨a, ha, _, ⟨b, hb, c, hc, habc.2, rfl⟩, hab.sup_right habc.1, (sup_assoc ..).symm⟩
  /-
    🎉 no goals
  -/


instance : @Std.Associative (Finset α) (· ○ ·) := ⟨disjSups_assoc⟩


theorem disjSups_left_comm : s ○ (t ○ u) = t ○ (s ○ u) := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t u : Finset α
    ⊢ Eq (s.disjSups (t.disjSups u)) (t.disjSups (s.disjSups u))
  -/
  simp_rw [← disjSups_assoc, disjSups_comm s]
  /-
    🎉 no goals
  -/


                                                          /-
                                                            α : Type u_2
                                                            inst✝³ : DecidableEq α
                                                            inst✝² : DistribLattice α
                                                            inst✝¹ : OrderBot α
                                                            inst✝ : DecidableRel Disjoint
                                                            s t u : Finset α
                                                            ⊢ Eq ((s.disjSups t).disjSups u) ((s.disjSups u).disjSups t)
                                                          -/
theorem disjSups_right_comm : s ○ t ○ u = s ○ u ○ t := by simp_rw [disjSups_assoc, disjSups_comm]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem disjSups_disjSups_disjSups_comm : s ○ t ○ (u ○ v) = s ○ u ○ (t ○ v) := by
  /-
    α : Type u_2
    inst✝³ : DecidableEq α
    inst✝² : DistribLattice α
    inst✝¹ : OrderBot α
    inst✝ : DecidableRel Disjoint
    s t u v : Finset α
    ⊢ Eq ((s.disjSups t).disjSups (u.disjSups v)) ((s.disjSups u).disjSups (t.disj …
  -/
  simp_rw [← disjSups_assoc, disjSups_right_comm]
  /-
    🎉 no goals
  -/


/-- `s \\ t` is the finset of elements of the form `a \ b` where `a ∈ s`, `b ∈ t`. -/
def diffs : Finset α → Finset α → Finset α := image₂ (· \ ·)


@[inherit_doc]
scoped[FinsetFamily] infixl:74 " \\\\ " => Finset.diffs
  -- This notation is meant to have higher precedence than `\` and `⊓`, but still within the
  -- realm of other binary notation


                                                                         /-
                                                                           α : Type u_2
                                                                           inst✝¹ : DecidableEq α
                                                                           inst✝ : GeneralizedBooleanAlgebra α
                                                                           s t : Finset α
                                                                           c : α
                                                                           ⊢ Iff (Membership.mem (s.diffs t) c) (Exists fun a => And (Membership.mem s a) …
                                                                         -/
@[simp] lemma mem_diffs : c ∈ s \\ t ↔ ∃ a ∈ s, ∃ b ∈ t, a \ b = c := by simp [(· \\ ·)]
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp, norm_cast] lemma coe_diffs : (↑(s \\ t) : Set α) = Set.image2 (· \ ·) s t :=
  coe_image₂ _ _ _


lemma card_diffs_le : #(s \\ t) ≤ #s * #t := card_image₂_le _ _ _


lemma card_diffs_iff : #(s \\ t) = #s * #t ↔ (s ×ˢ t : Set (α × α)).InjOn fun x ↦ x.1 \ x.2 :=
  card_image₂_iff


lemma sdiff_mem_diffs : a ∈ s → b ∈ t → a \ b ∈ s \\ t := mem_image₂_of_mem


lemma diffs_subset : s₁ ⊆ s₂ → t₁ ⊆ t₂ → s₁ \\ t₁ ⊆ s₂ \\ t₂ := image₂_subset

lemma diffs_subset_left : t₁ ⊆ t₂ → s \\ t₁ ⊆ s \\ t₂ := image₂_subset_left

lemma diffs_subset_right : s₁ ⊆ s₂ → s₁ \\ t ⊆ s₂ \\ t := image₂_subset_right


lemma image_subset_diffs_left : b ∈ t → s.image (· \ b) ⊆ s \\ t := image_subset_image₂_left


lemma image_subset_diffs_right : a ∈ s → t.image (a \ ·) ⊆ s \\ t := image_subset_image₂_right


lemma forall_mem_diffs {p : α → Prop} : (∀ c ∈ s \\ t, p c) ↔ ∀ a ∈ s, ∀ b ∈ t, p (a \ b) :=
  forall_mem_image₂


@[simp] lemma diffs_subset_iff : s \\ t ⊆ u ↔ ∀ a ∈ s, ∀ b ∈ t, a \ b ∈ u := image₂_subset_iff


@[simp]
lemma diffs_nonempty : (s \\ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty := image₂_nonempty_iff


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected lemma Nonempty.diffs : s.Nonempty → t.Nonempty → (s \\ t).Nonempty := Nonempty.image₂


lemma Nonempty.of_diffs_left : (s \\ t).Nonempty → s.Nonempty := Nonempty.of_image₂_left

lemma Nonempty.of_diffs_right : (s \\ t).Nonempty → t.Nonempty := Nonempty.of_image₂_right


@[simp] lemma empty_diffs : ∅ \\ t = ∅ := image₂_empty_left

@[simp] lemma diffs_empty : s \\ ∅ = ∅ := image₂_empty_right

@[simp] lemma diffs_eq_empty : s \\ t = ∅ ↔ s = ∅ ∨ t = ∅ := image₂_eq_empty_iff


@[simp] lemma singleton_diffs : {a} \\ t = t.image (a \ ·) := image₂_singleton_left

@[simp] lemma diffs_singleton : s \\ {b} = s.image (· \ b) := image₂_singleton_right

lemma singleton_diffs_singleton : ({a} \\ {b} : Finset α) = {a \ b} := image₂_singleton


lemma diffs_union_left : (s₁ ∪ s₂) \\ t = s₁ \\ t ∪ s₂ \\ t := image₂_union_left

lemma diffs_union_right : s \\ (t₁ ∪ t₂) = s \\ t₁ ∪ s \\ t₂ := image₂_union_right


lemma diffs_inter_subset_left : (s₁ ∩ s₂) \\ t ⊆ s₁ \\ t ∩ s₂ \\ t := image₂_inter_subset_left

lemma diffs_inter_subset_right : s \\ (t₁ ∩ t₂) ⊆ s \\ t₁ ∩ s \\ t₂ := image₂_inter_subset_right


lemma subset_diffs {s t : Set α} :
    ↑u ⊆ Set.image2 (· \ ·) s t → ∃ s' t' : Finset α, ↑s' ⊆ s ∧ ↑t' ⊆ t ∧ u ⊆ s' \\ t' :=
  subset_set_image₂


lemma biUnion_image_sdiff_left : s.biUnion (fun a ↦ t.image (a \ ·)) = s \\ t := biUnion_image_left

lemma biUnion_image_sdiff_right : t.biUnion (fun b ↦ s.image (· \ b)) = s \\ t :=
  biUnion_image_right


lemma image_sdiff_product (s t : Finset α) : (s ×ˢ t).image (uncurry (· \ ·)) = s \\ t :=
  image_uncurry_product _ _ _


lemma diffs_right_comm : s \\ t \\ u = s \\ u \\ t := image₂_right_comm sdiff_right_comm


/-- `sᶜˢ` is the finset of elements of the form `aᶜ` where `a ∈ s`. -/
def compls : Finset α → Finset α := map ⟨compl, compl_injective⟩


@[inherit_doc]
scoped[FinsetFamily] postfix:max "ᶜˢ" => Finset.compls


@[simp] lemma mem_compls : a ∈ sᶜˢ ↔ aᶜ ∈ s := by
  /-
    α : Type u_2
    inst✝ : BooleanAlgebra α
    s : Finset α
    a : α
    ⊢ Iff (Membership.mem s.compls a) (Membership.mem s (HasCompl.compl a))
  -/
  rw [Iff.comm, ← mem_map' ⟨compl, compl_injective⟩, Embedding.coeFn_mk, compl_compl, compls]
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        α : Type u_2
                                                                        inst✝¹ : BooleanAlgebra α
                                                                        s : Finset α
                                                                        inst✝ : DecidableEq α
                                                                        ⊢ Eq (Finset.image HasCompl.compl s) s.compls
                                                                      -/
@[simp] lemma image_compl [DecidableEq α] : s.image compl = sᶜˢ := by simp [compls, map_eq_image]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[simp, norm_cast] lemma coe_compls : (↑sᶜˢ : Set α) = compl '' ↑s := coe_map _ _


@[simp] lemma card_compls : #sᶜˢ = #s := card_map _


lemma compl_mem_compls : a ∈ s → aᶜ ∈ sᶜˢ := mem_map_of_mem _

@[simp] lemma compls_subset_compls : s₁ᶜˢ ⊆ s₂ᶜˢ ↔ s₁ ⊆ s₂ := map_subset_map

lemma forall_mem_compls {p : α → Prop} : (∀ a ∈ sᶜˢ, p a) ↔ ∀ a ∈ s, p aᶜ := forall_mem_map

                                                                                /-
                                                                                  α : Type u_2
                                                                                  inst✝ : BooleanAlgebra α
                                                                                  s : Finset α
                                                                                  p : α → Prop
                                                                                  ⊢ Iff (Exists fun a => And (Membership.mem s.compls a) (p a)) (Exists fun a => …
                                                                                -/
lemma exists_compls_iff {p : α → Prop} : (∃ a ∈ sᶜˢ, p a) ↔ ∃ a ∈ s, p aᶜ := by aesop
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


                                                             /-
                                                               α : Type u_2
                                                               inst✝ : BooleanAlgebra α
                                                               s : Finset α
                                                               ⊢ Eq s.compls.compls s
                                                             -/
@[simp] lemma compls_compls (s : Finset α) : sᶜˢᶜˢ = s := by ext; simp
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


                                                  /-
                                                    α : Type u_2
                                                    inst✝ : BooleanAlgebra α
                                                    s t : Finset α
                                                    ⊢ Iff (HasSubset.Subset s.compls t) (HasSubset.Subset s t.compls)
                                                  -/
lemma compls_subset_iff : sᶜˢ ⊆ t ↔ s ⊆ tᶜˢ := by rw [← compls_subset_compls, compls_compls]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
lemma compls_nonempty : sᶜˢ.Nonempty ↔ s.Nonempty := map_nonempty


protected alias ⟨Nonempty.of_compls, Nonempty.compls⟩ := compls_nonempty

@[simp] lemma compls_empty : (∅ : Finset α)ᶜˢ = ∅ := map_empty _

@[simp] lemma compls_eq_empty : sᶜˢ = ∅ ↔ s = ∅ := map_eq_empty

@[simp] lemma compls_singleton (a : α) : {a}ᶜˢ = {aᶜ} := map_singleton _ _

                                                                         /-
                                                                           α : Type u_2
                                                                           inst✝¹ : BooleanAlgebra α
                                                                           inst✝ : Fintype α
                                                                           ⊢ Eq Finset.univ.compls Finset.univ
                                                                         -/
@[simp] lemma compls_univ [Fintype α] : (univ : Finset α)ᶜˢ = univ := by ext; simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


@[simp] lemma compls_union (s t : Finset α) : (s ∪ t)ᶜˢ = sᶜˢ ∪ tᶜˢ := map_union _ _

@[simp] lemma compls_inter (s t : Finset α) : (s ∩ t)ᶜˢ = sᶜˢ ∩ tᶜˢ := map_inter _ _


@[simp] lemma compls_infs (s t : Finset α) : (s ⊼ t)ᶜˢ = sᶜˢ ⊻ tᶜˢ := by
  /-
    α : Type u_2
    inst✝¹ : BooleanAlgebra α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HasInfs.infs s t).compls (HasSups.sups s.compls t.compls)
  -/
  simp_rw [← image_compl]; exact image_image₂_distrib fun _ _ ↦ compl_inf
                           /-
                             🎉 no goals
                           -/


@[simp] lemma compls_sups (s t : Finset α) : (s ⊻ t)ᶜˢ = sᶜˢ ⊼ tᶜˢ := by
  /-
    α : Type u_2
    inst✝¹ : BooleanAlgebra α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HasSups.sups s t).compls (HasInfs.infs s.compls t.compls)
  -/
  simp_rw [← image_compl]; exact image_image₂_distrib fun _ _ ↦ compl_sup
                           /-
                             🎉 no goals
                           -/


@[simp] lemma infs_compls_eq_diffs (s t : Finset α) : s ⊼ tᶜˢ = s \\ t := by
  /-
    α : Type u_2
    inst✝¹ : BooleanAlgebra α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HasInfs.infs s t.compls) (s.diffs t)
  -/
  ext; simp [sdiff_eq]; aesop
                        /-
                          🎉 no goals
                        -/


@[simp] lemma compls_infs_eq_diffs (s t : Finset α) : sᶜˢ ⊼ t = t \\ s := by
  /-
    α : Type u_2
    inst✝¹ : BooleanAlgebra α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (HasInfs.infs s.compls t) (t.diffs s)
  -/
  rw [infs_comm, infs_compls_eq_diffs]
  /-
    🎉 no goals
  -/


@[simp] lemma diffs_compls_eq_infs (s t : Finset α) : s \\ tᶜˢ = s ⊼ t := by
  /-
    α : Type u_2
    inst✝¹ : BooleanAlgebra α
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (s.diffs t.compls) (HasInfs.infs s t)
  -/
  rw [← infs_compls_eq_diffs, compls_compls]
  /-
    🎉 no goals
  -/


protected lemma _root_.Set.Sized.compls (h𝒜 : (𝒜 : Set (Finset α)).Sized n) :
    (𝒜ᶜˢ : Set (Finset α)).Sized (Fintype.card α - n) :=
                                              /-
                                                α : Type u_4
                                                inst✝¹ : DecidableEq α
                                                inst✝ : Fintype α
                                                𝒜 : Finset (Finset α)
                                                n : Nat
                                                h𝒜 : Set.Sized n ↑𝒜
                                                s : Finset α
                                                hs : Membership.mem 𝒜 s
                                                ⊢ Eq (HasCompl.compl s).card (HSub.hSub (Fintype.card α) n)
                                              -/
  Finset.forall_mem_compls.2 <| fun s hs ↦ by rw [Finset.card_compl, h𝒜 hs]
                                              /-
                                                🎉 no goals
                                              -/


lemma sized_compls (hn : n ≤ Fintype.card α) :
    (𝒜ᶜˢ : Set (Finset α)).Sized n ↔ (𝒜 : Set (Finset α)).Sized (Fintype.card α - n) where
              /-
                α : Type u_4
                inst✝¹ : DecidableEq α
                inst✝ : Fintype α
                𝒜 : Finset (Finset α)
                n : Nat
                hn : LE.le n (Fintype.card α)
                h𝒜 : Set.Sized n ↑𝒜.compls
                ⊢ Set.Sized (HSub.hSub (Fintype.card α) n) ↑𝒜
              -/
  mp h𝒜 := by simpa using h𝒜.compls
              /-
                🎉 no goals
              -/
               /-
                 α : Type u_4
                 inst✝¹ : DecidableEq α
                 inst✝ : Fintype α
                 𝒜 : Finset (Finset α)
                 n : Nat
                 hn : LE.le n (Fintype.card α)
                 h𝒜 : Set.Sized (HSub.hSub (Fintype.card α) n) ↑𝒜
                 ⊢ Set.Sized n ↑𝒜.compls
               -/
  mpr h𝒜 := by simpa only [Nat.sub_sub_self hn] using h𝒜.compls
               /-
                 🎉 no goals
               -/


