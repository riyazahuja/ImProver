theorem OrdConnected.out (h : OrdConnected s) : ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s), Icc x y ⊆ s :=
  h.1


theorem ordConnected_def : OrdConnected s ↔ ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s), Icc x y ⊆ s :=
  ⟨fun h => h.1, fun h => ⟨h⟩⟩


/-- It suffices to prove `[[x, y]] ⊆ s` for `x y ∈ s`, `x ≤ y`. -/
theorem ordConnected_iff : OrdConnected s ↔ ∀ x ∈ s, ∀ y ∈ s, x ≤ y → Icc x y ⊆ s :=
  ordConnected_def.trans
    ⟨fun hs _ hx _ hy _ => hs hx hy, fun H x hx y hy _ hz => H x hx y hy (le_trans hz.1 hz.2) hz⟩


theorem ordConnected_of_Ioo {α : Type*} [PartialOrder α] {s : Set α}
    (hs : ∀ x ∈ s, ∀ y ∈ s, x < y → Ioo x y ⊆ s) : OrdConnected s := by
  /-
    α : Type u_3
    inst✝ : PartialOrder α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y …
    ⊢ s.OrdConnected
  -/
  rw [ordConnected_iff]
  /-
    α : Type u_3
    inst✝ : PartialOrder α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y …
    ⊢ ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LE.le x y →  …
  -/
  intro x hx y hy hxy
  /-
    α : Type u_3
    inst✝ : PartialOrder α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y …
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : LE.le x y
    ⊢ HasSubset.Subset (Set.Icc x y) s
  -/
  rcases eq_or_lt_of_le hxy with (rfl | hxy'); · simpa
                                                 /-
                                                   🎉 no goals
                                                 -/
  /-
    case inr
    α : Type u_3
    inst✝ : PartialOrder α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y …
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : LE.le x y
    hxy' : LT.lt x y
    ⊢ HasSubset.Subset (Set.Icc x y) s
  -/
  rw [← Ioc_insert_left hxy, ← Ioo_insert_right hxy']
  /-
    case inr
    α : Type u_3
    inst✝ : PartialOrder α
    s : Set α
    hs : ∀ (x : α), Membership.mem s x → ∀ (y : α), Membership.mem s y → LT.lt x y …
    x : α
    hx : Membership.mem s x
    y : α
    hy : Membership.mem s y
    hxy : LE.le x y
    hxy' : LT.lt x y
    ⊢ HasSubset.Subset (Insert.insert x (Insert.insert y (Set.Ioo x y))) s
  -/
  exact insert_subset_iff.2 ⟨hx, insert_subset_iff.2 ⟨hy, hs x hx y hy hxy'⟩⟩
  /-
    🎉 no goals
  -/


theorem OrdConnected.preimage_mono {f : β → α} (hs : OrdConnected s) (hf : Monotone f) :
    OrdConnected (f ⁻¹' s) :=
  ⟨fun _ hx _ hy _ hz => hs.out hx hy ⟨hf hz.1, hf hz.2⟩⟩


theorem OrdConnected.preimage_anti {f : β → α} (hs : OrdConnected s) (hf : Antitone f) :
    OrdConnected (f ⁻¹' s) :=
  ⟨fun _ hx _ hy _ hz => hs.out hy hx ⟨hf hz.2, hf hz.1⟩⟩


protected theorem Icc_subset (s : Set α) [hs : OrdConnected s] {x y} (hx : x ∈ s) (hy : y ∈ s) :
    Icc x y ⊆ s :=
  hs.out hx hy


theorem image_Icc (e : α ↪o β) (he : OrdConnected (range e)) (x y : α) :
    e '' Icc x y = Icc (e x) (e y) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : Preorder α
    inst✝ : Preorder β
    e : OrderEmbedding α β
    he : (Set.range ⇑e).OrdConnected
    x y : α
    ⊢ Eq (Set.image (⇑e) (Set.Icc x y)) (Set.Icc (e x) (e y))
  -/
  rw [← e.preimage_Icc, image_preimage_eq_inter_range, inter_eq_left.2 (he.out ⟨_, rfl⟩ ⟨_, rfl⟩)]
  /-
    🎉 no goals
  -/


theorem image_Ico (e : α ↪o β) (he : OrdConnected (range e)) (x y : α) :
    e '' Ico x y = Ico (e x) (e y) := by
  rw [← e.preimage_Ico, image_preimage_eq_inter_range,
    inter_eq_left.2 <| Ico_subset_Icc_self.trans <| he.out ⟨_, rfl⟩ ⟨_, rfl⟩]


theorem image_Ioc (e : α ↪o β) (he : OrdConnected (range e)) (x y : α) :
    e '' Ioc x y = Ioc (e x) (e y) := by
  rw [← e.preimage_Ioc, image_preimage_eq_inter_range,
    inter_eq_left.2 <| Ioc_subset_Icc_self.trans <| he.out ⟨_, rfl⟩ ⟨_, rfl⟩]


theorem image_Ioo (e : α ↪o β) (he : OrdConnected (range e)) (x y : α) :
    e '' Ioo x y = Ioo (e x) (e y) := by
  rw [← e.preimage_Ioo, image_preimage_eq_inter_range,
    inter_eq_left.2 <| Ioo_subset_Icc_self.trans <| he.out ⟨_, rfl⟩ ⟨_, rfl⟩]


@[simp]
lemma image_subtype_val_Icc {s : Set α} [OrdConnected s] (x y : s) :
    Subtype.val '' Icc x y = Icc x.1 y :=
                                                 /-
                                                   α : Type u_1
                                                   inst✝¹ : Preorder α
                                                   s : Set α
                                                   inst✝ : s.OrdConnected
                                                   x y : ↑s
                                                   ⊢ (Set.range ⇑(OrderEmbedding.subtype fun x => Membership.mem s x)).OrdConnected
                                                 -/
  (OrderEmbedding.subtype (· ∈ s)).image_Icc (by simpa) x y
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
lemma image_subtype_val_Ico {s : Set α} [OrdConnected s] (x y : s) :
    Subtype.val '' Ico x y = Ico x.1 y :=
                                                 /-
                                                   α : Type u_1
                                                   inst✝¹ : Preorder α
                                                   s : Set α
                                                   inst✝ : s.OrdConnected
                                                   x y : ↑s
                                                   ⊢ (Set.range ⇑(OrderEmbedding.subtype fun x => Membership.mem s x)).OrdConnected
                                                 -/
  (OrderEmbedding.subtype (· ∈ s)).image_Ico (by simpa) x y
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
lemma image_subtype_val_Ioc {s : Set α} [OrdConnected s] (x y : s) :
    Subtype.val '' Ioc x y = Ioc x.1 y :=
                                                 /-
                                                   α : Type u_1
                                                   inst✝¹ : Preorder α
                                                   s : Set α
                                                   inst✝ : s.OrdConnected
                                                   x y : ↑s
                                                   ⊢ (Set.range ⇑(OrderEmbedding.subtype fun x => Membership.mem s x)).OrdConnected
                                                 -/
  (OrderEmbedding.subtype (· ∈ s)).image_Ioc (by simpa) x y
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
lemma image_subtype_val_Ioo {s : Set α} [OrdConnected s] (x y : s) :
    Subtype.val '' Ioo x y = Ioo x.1 y :=
                                                 /-
                                                   α : Type u_1
                                                   inst✝¹ : Preorder α
                                                   s : Set α
                                                   inst✝ : s.OrdConnected
                                                   x y : ↑s
                                                   ⊢ (Set.range ⇑(OrderEmbedding.subtype fun x => Membership.mem s x)).OrdConnected
                                                 -/
  (OrderEmbedding.subtype (· ∈ s)).image_Ioo (by simpa) x y
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem OrdConnected.inter {s t : Set α} (hs : OrdConnected s) (ht : OrdConnected t) :
    OrdConnected (s ∩ t) :=
  ⟨fun _ hx _ hy => subset_inter (hs.out hx.1 hy.1) (ht.out hx.2 hy.2)⟩


instance OrdConnected.inter' {s t : Set α} [OrdConnected s] [OrdConnected t] :
    OrdConnected (s ∩ t) :=
  OrdConnected.inter ‹_› ‹_›


theorem OrdConnected.dual {s : Set α} (hs : OrdConnected s) :
    OrdConnected (OrderDual.ofDual ⁻¹' s) :=
  ⟨fun _ hx _ hy _ hz => hs.out hy hx ⟨hz.2, hz.1⟩⟩


theorem ordConnected_dual {s : Set α} : OrdConnected (OrderDual.ofDual ⁻¹' s) ↔ OrdConnected s :=
               /-
                 α : Type u_1
                 inst✝ : Preorder α
                 s : Set α
                 h : (Set.preimage (⇑OrderDual.ofDual) s).OrdConnected
                 ⊢ s.OrdConnected
               -/
  ⟨fun h => by simpa only [ordConnected_def] using h.dual, fun h => h.dual⟩
               /-
                 🎉 no goals
               -/


theorem ordConnected_sInter {S : Set (Set α)} (hS : ∀ s ∈ S, OrdConnected s) :
    OrdConnected (⋂₀ S) :=
  ⟨fun _x hx _y hy _z hz s hs => (hS s hs).out (hx s hs) (hy s hs) hz⟩


theorem ordConnected_iInter {ι : Sort*} {s : ι → Set α} (hs : ∀ i, OrdConnected (s i)) :
    OrdConnected (⋂ i, s i) :=
  ordConnected_sInter <| forall_mem_range.2 hs


instance ordConnected_iInter' {ι : Sort*} {s : ι → Set α} [∀ i, OrdConnected (s i)] :
    OrdConnected (⋂ i, s i) :=
  ordConnected_iInter ‹_›

/- ./././Mathport/Syntax/Translate/Expr.lean:107:6: warning: expanding binder group (i hi) -/

theorem ordConnected_biInter {ι : Sort*} {p : ι → Prop} {s : ∀ i, p i → Set α}
    (hs : ∀ i hi, OrdConnected (s i hi)) : OrdConnected (⋂ (i) (hi), s i hi) :=
  ordConnected_iInter fun i => ordConnected_iInter <| hs i


theorem ordConnected_pi {ι : Type*} {α : ι → Type*} [∀ i, Preorder (α i)] {s : Set ι}
    {t : ∀ i, Set (α i)} (h : ∀ i ∈ s, OrdConnected (t i)) : OrdConnected (s.pi t) :=
  ⟨fun _ hx _ hy _ hz i hi => (h i hi).out (hx i hi) (hy i hi) ⟨hz.1 i, hz.2 i⟩⟩


instance ordConnected_pi' {ι : Type*} {α : ι → Type*} [∀ i, Preorder (α i)] {s : Set ι}
    {t : ∀ i, Set (α i)} [h : ∀ i, OrdConnected (t i)] : OrdConnected (s.pi t) :=
  ordConnected_pi fun i _ => h i


@[instance]
theorem ordConnected_Ici {a : α} : OrdConnected (Ici a) :=
  ⟨fun _ hx _ _ _ hz => le_trans hx hz.1⟩


@[instance]
theorem ordConnected_Iic {a : α} : OrdConnected (Iic a) :=
  ⟨fun _ _ _ hy _ hz => le_trans hz.2 hy⟩


@[instance]
theorem ordConnected_Ioi {a : α} : OrdConnected (Ioi a) :=
  ⟨fun _ hx _ _ _ hz => lt_of_lt_of_le hx hz.1⟩


@[instance]
theorem ordConnected_Iio {a : α} : OrdConnected (Iio a) :=
  ⟨fun _ _ _ hy _ hz => lt_of_le_of_lt hz.2 hy⟩


@[instance]
theorem ordConnected_Icc {a b : α} : OrdConnected (Icc a b) :=
  ordConnected_Ici.inter ordConnected_Iic


@[instance]
theorem ordConnected_Ico {a b : α} : OrdConnected (Ico a b) :=
  ordConnected_Ici.inter ordConnected_Iio


@[instance]
theorem ordConnected_Ioc {a b : α} : OrdConnected (Ioc a b) :=
  ordConnected_Ioi.inter ordConnected_Iic


@[instance]
theorem ordConnected_Ioo {a b : α} : OrdConnected (Ioo a b) :=
  ordConnected_Ioi.inter ordConnected_Iio


@[instance]
theorem ordConnected_singleton {α : Type*} [PartialOrder α] {a : α} :
    OrdConnected ({a} : Set α) := by
  /-
    α : Type u_3
    inst✝ : PartialOrder α
    a : α
    ⊢ (Singleton.singleton a).OrdConnected
  -/
  rw [← Icc_self]
  /-
    α : Type u_3
    inst✝ : PartialOrder α
    a : α
    ⊢ (Set.Icc a a).OrdConnected
  -/
  exact ordConnected_Icc
  /-
    🎉 no goals
  -/


@[instance]
theorem ordConnected_empty : OrdConnected (∅ : Set α) :=
  ⟨fun _ => False.elim⟩


@[instance]
theorem ordConnected_univ : OrdConnected (univ : Set α) :=
  ⟨fun _ _ _ _ => subset_univ _⟩


/-- In a dense order `α`, the subtype from an `OrdConnected` set is also densely ordered. -/
instance instDenselyOrdered [DenselyOrdered α] {s : Set α} [hs : OrdConnected s] :
    DenselyOrdered s :=
  ⟨fun a b (h : (a : α) < b) =>
    let ⟨x, H⟩ := exists_between h
    ⟨⟨x, (hs.out a.2 b.2) (Ioo_subset_Icc_self H)⟩, H⟩⟩


@[instance]
theorem ordConnected_preimage {F : Type*} [FunLike F α β] [OrderHomClass F α β] (f : F)
    {s : Set β} [hs : OrdConnected s] : OrdConnected (f ⁻¹' s) :=
  ⟨fun _ hx _ hy _ hz => hs.out hx hy ⟨OrderHomClass.mono _ hz.1, OrderHomClass.mono _ hz.2⟩⟩


@[instance]
theorem ordConnected_image {E : Type*} [EquivLike E α β] [OrderIsoClass E α β] (e : E) {s : Set α}
    [hs : OrdConnected s] : OrdConnected (e '' s) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Preorder β
    E : Type u_3
    inst✝¹ : EquivLike E α β
    inst✝ : OrderIsoClass E α β
    e : E
    s : Set α
    hs : s.OrdConnected
    ⊢ (Set.image (⇑e) s).OrdConnected
  -/
  erw [(e : α ≃o β).image_eq_preimage]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Preorder β
    E : Type u_3
    inst✝¹ : EquivLike E α β
    inst✝ : OrderIsoClass E α β
    e : E
    s : Set α
    hs : s.OrdConnected
    ⊢ (Set.preimage (⇑(↑e).symm) s).OrdConnected
  -/
  apply ordConnected_preimage (e : α ≃o β).symm
  /-
    🎉 no goals
  -/

-- Porting note: split up `simp_rw [← image_univ, OrdConnected_image e]`, would not work otherwise

@[instance]
theorem ordConnected_range {E : Type*} [EquivLike E α β] [OrderIsoClass E α β] (e : E) :
    OrdConnected (range e) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Preorder β
    E : Type u_3
    inst✝¹ : EquivLike E α β
    inst✝ : OrderIsoClass E α β
    e : E
    ⊢ (Set.range ⇑e).OrdConnected
  -/
  simp_rw [← image_univ]
  /-
    α : Type u_1
    β : Type u_2
    inst✝³ : Preorder α
    inst✝² : Preorder β
    E : Type u_3
    inst✝¹ : EquivLike E α β
    inst✝ : OrderIsoClass E α β
    e : E
    ⊢ (Set.image (⇑e) Set.univ).OrdConnected
  -/
  exact ordConnected_image (e : α ≃o β)
  /-
    🎉 no goals
  -/


@[simp]
theorem dual_ordConnected_iff {s : Set α} : OrdConnected (ofDual ⁻¹' s) ↔ OrdConnected s := by
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (Set.preimage (⇑OrderDual.ofDual) s).OrdConnected s.OrdConnected
  -/
  simp_rw [ordConnected_def, toDual.surjective.forall, dual_Icc, Subtype.forall']
  /-
    α : Type u_1
    inst✝ : Preorder α
    s : Set α
    ⊢ Iff (∀ (x x_1 : Subtype fun a => Membership.mem (Set.preimage (⇑OrderDual.of …
  -/
  exact forall_swap
  /-
    🎉 no goals
  -/


@[instance]
theorem dual_ordConnected {s : Set α} [OrdConnected s] : OrdConnected (ofDual ⁻¹' s) :=
  dual_ordConnected_iff.2 ‹_›


protected theorem _root_.IsAntichain.ordConnected (hs : IsAntichain (· ≤ ·) s) : s.OrdConnected :=
  ⟨fun x hx y hy z hz => by
    /-
      α : Type u_1
      inst✝ : PartialOrder α
      s : Set α
      hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
      x : α
      hx : Membership.mem s x
      y : α
      hy : Membership.mem s y
      z : α
      hz : Membership.mem (Set.Icc x y) z
      ⊢ Membership.mem s z
    -/
    obtain rfl := hs.eq hx hy (hz.1.trans hz.2)
    /-
      α : Type u_1
      inst✝ : PartialOrder α
      s : Set α
      hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
      x : α
      hx : Membership.mem s x
      z : α
      hy : Membership.mem s x
      hz : Membership.mem (Set.Icc x x) z
      ⊢ Membership.mem s z
    -/
    rw [Icc_self, mem_singleton_iff] at hz
    /-
      α : Type u_1
      inst✝ : PartialOrder α
      s : Set α
      hs : IsAntichain (fun x1 x2 => LE.le x1 x2) s
      x : α
      hx : Membership.mem s x
      z : α
      hy : Membership.mem s x
      hz : Eq z x
      ⊢ Membership.mem s z
    -/
    rwa [hz]⟩
    /-
      🎉 no goals
    -/


lemma ordConnected_inter_Icc_of_subset (h : Ioo x y ⊆ s) : OrdConnected (s ∩ Icc x y) :=
  ordConnected_of_Ioo fun _u ⟨_, hu, _⟩ _v ⟨_, _, hv⟩ _ ↦
    Ioo_subset_Ioo hu hv |>.trans <| subset_inter h Ioo_subset_Icc_self


lemma ordConnected_inter_Icc_iff (hx : x ∈ s) (hy : y ∈ s) :
    OrdConnected (s ∩ Icc x y) ↔ Ioo x y ⊆ s := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Iff (Inter.inter s (Set.Icc x y)).OrdConnected (HasSubset.Subset (Set.Ioo x  …
  -/
  refine ⟨fun h ↦ Ioo_subset_Icc_self.trans fun z hz ↦ ?_, ordConnected_inter_Icc_of_subset⟩
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    h : (Inter.inter s (Set.Icc x y)).OrdConnected
    z : α
    hz : Membership.mem (Set.Icc x y) z
    ⊢ Membership.mem s z
  -/
  have hxy : x ≤ y := hz.1.trans hz.2
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    h : (Inter.inter s (Set.Icc x y)).OrdConnected
    z : α
    hz : Membership.mem (Set.Icc x y) z
    hxy : LE.le x y
    ⊢ Membership.mem s z
  -/
  exact h.out ⟨hx, left_mem_Icc.2 hxy⟩ ⟨hy, right_mem_Icc.2 hxy⟩ hz |>.1
  /-
    🎉 no goals
  -/


lemma not_ordConnected_inter_Icc_iff (hx : x ∈ s) (hy : y ∈ s) :
    ¬ OrdConnected (s ∩ Icc x y) ↔ ∃ z ∉ s, z ∈ Ioo x y := by
  /-
    α : Type u_1
    inst✝ : PartialOrder α
    s : Set α
    x y : α
    hx : Membership.mem s x
    hy : Membership.mem s y
    ⊢ Iff (Not (Inter.inter s (Set.Icc x y)).OrdConnected) (Exists fun z => And (N …
  -/
  simp_rw [ordConnected_inter_Icc_iff hx hy, subset_def, not_forall, exists_prop, and_comm]
  /-
    🎉 no goals
  -/


@[instance]
theorem ordConnected_uIcc {a b : α} : OrdConnected [[a, b]] :=
  ordConnected_Icc


@[instance]
theorem ordConnected_uIoc {a b : α} : OrdConnected (Ι a b) :=
  ordConnected_Ioc


theorem OrdConnected.uIcc_subset (hs : OrdConnected s) ⦃x⦄ (hx : x ∈ s) ⦃y⦄ (hy : y ∈ s) :
    [[x, y]] ⊆ s :=
  hs.out (min_rec' (· ∈ s) hx hy) (max_rec' (· ∈ s) hx hy)


theorem OrdConnected.uIoc_subset (hs : OrdConnected s) ⦃x⦄ (hx : x ∈ s) ⦃y⦄ (hy : y ∈ s) :
    Ι x y ⊆ s :=
  Ioc_subset_Icc_self.trans <| hs.uIcc_subset hx hy


theorem ordConnected_iff_uIcc_subset :
    OrdConnected s ↔ ∀ ⦃x⦄ (_ : x ∈ s) ⦃y⦄ (_ : y ∈ s), [[x, y]] ⊆ s :=
  ⟨fun h => h.uIcc_subset, fun H => ⟨fun _ hx _ hy => Icc_subset_uIcc.trans <| H hx hy⟩⟩


theorem ordConnected_of_uIcc_subset_left (h : ∀ y ∈ s, [[x, y]] ⊆ s) : OrdConnected s :=
  ordConnected_iff_uIcc_subset.2 fun y hy z hz =>
    calc
      [[y, z]] ⊆ [[y, x]] ∪ [[x, z]] := uIcc_subset_uIcc_union_uIcc
                                    /-
                                      α : Type u_1
                                      inst✝ : LinearOrder α
                                      s : Set α
                                      x : α
                                      h : ∀ (y : α), Membership.mem s y → HasSubset.Subset (Set.uIcc x y) s
                                      y : α
                                      hy : Membership.mem s y
                                      z : α
                                      hz : Membership.mem s z
                                      ⊢ Eq (Union.union (Set.uIcc y x) (Set.uIcc x z)) (Union.union (Set.uIcc x y) ( …
                                    -/
      _ = [[x, y]] ∪ [[x, z]] := by rw [uIcc_comm]
                                    /-
                                      🎉 no goals
                                    -/
      _ ⊆ s := union_subset (h y hy) (h z hz)


theorem ordConnected_iff_uIcc_subset_left (hx : x ∈ s) :
    OrdConnected s ↔ ∀ ⦃y⦄, y ∈ s → [[x, y]] ⊆ s :=
  ⟨fun hs => hs.uIcc_subset hx, ordConnected_of_uIcc_subset_left⟩


theorem ordConnected_iff_uIcc_subset_right (hx : x ∈ s) :
    OrdConnected s ↔ ∀ ⦃y⦄, y ∈ s → [[y, x]] ⊆ s := by
  /-
    α : Type u_1
    inst✝ : LinearOrder α
    s : Set α
    x : α
    hx : Membership.mem s x
    ⊢ Iff s.OrdConnected (∀ ⦃y : α⦄, Membership.mem s y → HasSubset.Subset (Set.uI …
  -/
  simp_rw [ordConnected_iff_uIcc_subset_left hx, uIcc_comm]
  /-
    🎉 no goals
  -/


