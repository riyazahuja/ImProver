/-- When `f` is an embedding of `α` in `β` and `s` is a finset in `α`, then `s.map f` is the image
finset in `β`. The embedding condition guarantees that there are no duplicates in the image. -/
def map (f : α ↪ β) (s : Finset α) : Finset β :=
  ⟨s.1.map f, s.2.map f.2⟩


@[simp]
theorem map_val (f : α ↪ β) (s : Finset α) : (map f s).1 = s.1.map f :=
  rfl


@[simp]
theorem map_empty (f : α ↪ β) : (∅ : Finset α).map f = ∅ :=
  rfl


@[simp]
theorem mem_map {b : β} : b ∈ s.map f ↔ ∃ a ∈ s, f a = b :=
  Multiset.mem_map

-- Porting note: Higher priority to apply before `mem_map`.

@[simp 1100]
theorem mem_map_equiv {f : α ≃ β} {b : β} : b ∈ s.map f.toEmbedding ↔ f.symm b ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : Equiv α β
    b : β
    ⊢ Iff (Membership.mem (Finset.map f.toEmbedding s) b) (Membership.mem s (f.sym …
  -/
  rw [mem_map]
  exact
    ⟨by
      rintro ⟨a, H, rfl⟩
      simpa, fun h => ⟨_, h, by simp⟩⟩


@[simp 1100]
theorem mem_map' (f : α ↪ β) {a} {s : Finset α} : f a ∈ s.map f ↔ a ∈ s :=
  mem_map_of_injective f.2


theorem mem_map_of_mem (f : α ↪ β) {a} {s : Finset α} : a ∈ s → f a ∈ s.map f :=
  (mem_map' _).2


theorem forall_mem_map {f : α ↪ β} {s : Finset α} {p : ∀ a, a ∈ s.map f → Prop} :
    (∀ y (H : y ∈ s.map f), p y H) ↔ ∀ x (H : x ∈ s), p (f x) (mem_map_of_mem _ H) :=
  ⟨fun h y hy => h (f y) (mem_map_of_mem _ hy),
   fun h x hx => by
    /-
      α : Type u_1
      β : Type u_2
      f : Function.Embedding α β
      s : Finset α
      p : (a : β) → Membership.mem (Finset.map f s) a → Prop
      h : ∀ (x : α) (H : Membership.mem s x), p (f x) ⋯
      x : β
      hx : Membership.mem (Finset.map f s) x
      ⊢ p x hx
    -/
    obtain ⟨y, hy, rfl⟩ := mem_map.1 hx
    /-
      case intro.intro
      α : Type u_1
      β : Type u_2
      f : Function.Embedding α β
      s : Finset α
      p : (a : β) → Membership.mem (Finset.map f s) a → Prop
      h : ∀ (x : α) (H : Membership.mem s x), p (f x) ⋯
      y : α
      hy : Membership.mem s y
      hx : Membership.mem (Finset.map f s) (f y)
      ⊢ p (f y) hx
    -/
    exact h _ hy⟩
    /-
      🎉 no goals
    -/


theorem apply_coe_mem_map (f : α ↪ β) (s : Finset α) (x : s) : f x ∈ s.map f :=
  mem_map_of_mem f x.prop


@[simp, norm_cast]
theorem coe_map (f : α ↪ β) (s : Finset α) : (s.map f : Set β) = f '' s :=
              /-
                α : Type u_1
                β : Type u_2
                f : Function.Embedding α β
                s : Finset α
                ⊢ ∀ (x : β), Iff (Membership.mem (↑(Finset.map f s)) x) (Membership.mem (Set.i …
              -/
  Set.ext (by simp only [mem_coe, mem_map, Set.mem_image, implies_true])
              /-
                🎉 no goals
              -/


theorem coe_map_subset_range (f : α ↪ β) (s : Finset α) : (s.map f : Set β) ⊆ Set.range f :=
  calc
    ↑(s.map f) = f '' s := coe_map f s
    _ ⊆ Set.range f := Set.image_subset_range f ↑s


/-- If the only elements outside `s` are those left fixed by `σ`, then mapping by `σ` has no effect.
-/
theorem map_perm {σ : Equiv.Perm α} (hs : { a | σ a ≠ a } ⊆ s) : s.map (σ : α ↪ α) = s :=
  coe_injective <| (coe_map _ _).trans <| Set.image_perm hs


theorem map_toFinset [DecidableEq α] [DecidableEq β] {s : Multiset α} :
    s.toFinset.map f = (s.map f).toFinset :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    f : Function.Embedding α β
                    inst✝¹ : DecidableEq α
                    inst✝ : DecidableEq β
                    s : Multiset α
                    x✝ : β
                    ⊢ Iff (Membership.mem (Finset.map f s.toFinset) x✝) (Membership.mem (Multiset. …
                  -/
  ext fun _ => by simp only [mem_map, Multiset.mem_map, exists_prop, Multiset.mem_toFinset]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem map_refl : s.map (Embedding.refl _) = s :=
                  /-
                    α : Type u_1
                    s : Finset α
                    x✝ : α
                    ⊢ Iff (Membership.mem (Finset.map (Function.Embedding.refl α) s) x✝) (Membersh …
                  -/
  ext fun _ => by simpa only [mem_map, exists_prop] using exists_eq_right
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem map_cast_heq {α β} (h : α = β) (s : Finset α) :
    HEq (s.map (Equiv.cast h).toEmbedding) s := by
  /-
    α β : Type u_4
    h : Eq α β
    s : Finset α
    ⊢ HEq (Finset.map (Equiv.cast h).toEmbedding s) s
  -/
  subst h
  /-
    α : Type u_4
    s : Finset α
    ⊢ HEq (Finset.map (Equiv.cast ⋯).toEmbedding s) s
  -/
  simp
  /-
    🎉 no goals
  -/


theorem map_map (f : α ↪ β) (g : β ↪ γ) (s : Finset α) : (s.map f).map g = s.map (f.trans g) :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    f : Function.Embedding α β
                    g : Function.Embedding β γ
                    s : Finset α
                    ⊢ Eq (Finset.map g (Finset.map f s)).val (Finset.map (f.trans g) s).val
                  -/
  eq_of_veq <| by simp only [map_val, Multiset.map_map]; rfl
                                                         /-
                                                           🎉 no goals
                                                         -/


theorem map_comm {β'} {f : β ↪ γ} {g : α ↪ β} {f' : α ↪ β'} {g' : β' ↪ γ}
    (h_comm : ∀ a, f (g a) = g' (f' a)) : (s.map g).map f = (s.map f').map g' := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    s : Finset α
    β' : Type u_4
    f : Function.Embedding β γ
    g : Function.Embedding α β
    f' : Function.Embedding α β'
    g' : Function.Embedding β' γ
    h_comm : ∀ (a : α), Eq (f (g a)) (g' (f' a))
    ⊢ Eq (Finset.map f (Finset.map g s)) (Finset.map g' (Finset.map f' s))
  -/
  simp_rw [map_map, Embedding.trans, Function.comp_def, h_comm]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Semiconj.finset_map {f : α ↪ β} {ga : α ↪ α} {gb : β ↪ β}
    (h : Function.Semiconj f ga gb) : Function.Semiconj (map f) (map ga) (map gb) := fun _ =>
  map_comm h


theorem _root_.Function.Commute.finset_map {f g : α ↪ α} (h : Function.Commute f g) :
    Function.Commute (map f) (map g) :=
  Function.Semiconj.finset_map h


@[simp]
theorem map_subset_map {s₁ s₂ : Finset α} : s₁.map f ⊆ s₂.map f ↔ s₁ ⊆ s₂ :=
  ⟨fun h _ xs => (mem_map' _).1 <| h <| (mem_map' f).2 xs,
               /-
                 α : Type u_1
                 β : Type u_2
                 f : Function.Embedding α β
                 s₁ s₂ : Finset α
                 h : HasSubset.Subset s₁ s₂
                 ⊢ HasSubset.Subset (Finset.map f s₁) (Finset.map f s₂)
               -/
   fun h => by simp [subset_def, Multiset.map_subset_map h]⟩
               /-
                 🎉 no goals
               -/


@[gcongr] alias ⟨_, _root_.GCongr.finsetMap_subset⟩ := map_subset_map


/-- The `Finset` version of `Equiv.subset_symm_image`. -/
theorem subset_map_symm {t : Finset β} {f : α ≃ β} : s ⊆ t.map f.symm ↔ s.map f ⊆ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    f : Equiv α β
    ⊢ Iff (HasSubset.Subset s (Finset.map f.symm.toEmbedding t)) (HasSubset.Subset …
  -/
  constructor <;> intro h x hx
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s : Finset α
      t : Finset β
      f : Equiv α β
      h : HasSubset.Subset s (Finset.map f.symm.toEmbedding t)
      x : β
      hx : Membership.mem (Finset.map f.toEmbedding s) x
      ⊢ Membership.mem t x
    -/
  · simp only [mem_map_equiv, Equiv.symm_symm] at hx
    /-
      case mp
      α : Type u_1
      β : Type u_2
      s : Finset α
      t : Finset β
      f : Equiv α β
      h : HasSubset.Subset s (Finset.map f.symm.toEmbedding t)
      x : β
      hx : Membership.mem s (f.symm x)
      ⊢ Membership.mem t x
    -/
    simpa using h hx
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      s : Finset α
      t : Finset β
      f : Equiv α β
      h : HasSubset.Subset (Finset.map f.toEmbedding s) t
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem (Finset.map f.symm.toEmbedding t) x
    -/
  · simp only [mem_map_equiv]
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      s : Finset α
      t : Finset β
      f : Equiv α β
      h : HasSubset.Subset (Finset.map f.toEmbedding s) t
      x : α
      hx : Membership.mem s x
      ⊢ Membership.mem t (f.symm.symm x)
    -/
    exact h (by simp [hx])
    /-
      🎉 no goals
    -/


/-- The `Finset` version of `Equiv.symm_image_subset`. -/
theorem map_symm_subset {t : Finset β} {f : α ≃ β} : t.map f.symm ⊆ s ↔ t ⊆ s.map f := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    f : Equiv α β
    ⊢ Iff (HasSubset.Subset (Finset.map f.symm.toEmbedding t) s) (HasSubset.Subset …
  -/
  simp only [← subset_map_symm, Equiv.symm_symm]
  /-
    🎉 no goals
  -/


/-- Associate to an embedding `f` from `α` to `β` the order embedding that maps a finset to its
image under `f`. -/
def mapEmbedding (f : α ↪ β) : Finset α ↪o Finset β :=
  OrderEmbedding.ofMapLEIff (map f) fun _ _ => map_subset_map


@[simp]
theorem map_inj {s₁ s₂ : Finset α} : s₁.map f = s₂.map f ↔ s₁ = s₂ :=
  (mapEmbedding f).injective.eq_iff


theorem map_injective (f : α ↪ β) : Injective (map f) :=
  (mapEmbedding f).injective


@[simp]
theorem map_ssubset_map {s t : Finset α} : s.map f ⊂ t.map f ↔ s ⊂ t := (mapEmbedding f).lt_iff_lt


@[gcongr] alias ⟨_, _root_.GCongr.finsetMap_ssubset⟩ := map_ssubset_map


@[simp]
theorem mapEmbedding_apply : mapEmbedding f s = map f s :=
  rfl


theorem filter_map {p : β → Prop} [DecidablePred p] :
    (s.map f).filter p = (s.filter (p ∘ f)).map f :=
  eq_of_veq (Multiset.filter_map _ _ _)


lemma map_filter' (p : α → Prop) [DecidablePred p] (f : α ↪ β) (s : Finset α)
    [DecidablePred (∃ a, p a ∧ f a = ·)] :
    (s.filter p).map f = (s.map f).filter fun b => ∃ a, p a ∧ f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    p : α → Prop
    inst✝¹ : DecidablePred p
    f : Function.Embedding α β
    s : Finset α
    inst✝ : DecidablePred fun x => Exists fun a => And (p a) (Eq (f a) x)
    ⊢ Eq (Finset.map f (Finset.filter p s)) (Finset.filter (fun b => Exists fun a  …
  -/
  simp [Function.comp_def, filter_map, f.injective.eq_iff]
  /-
    🎉 no goals
  -/


lemma filter_attach' [DecidableEq α] (s : Finset α) (p : s → Prop) [DecidablePred p] :
    s.attach.filter p =
      (s.filter fun x => ∃ h, p ⟨x, h⟩).attach.map
        ⟨Subtype.map id <| filter_subset _ _, Subtype.map_injective _ injective_id⟩ :=
  eq_of_veq <| Multiset.filter_attach' _ _


lemma filter_attach (p : α → Prop) [DecidablePred p] (s : Finset α) :
    s.attach.filter (fun a : s ↦ p a) =
      (s.filter p).attach.map ((Embedding.refl _).subtypeMap mem_of_mem_filter) :=
  eq_of_veq <| Multiset.filter_attach _ _


theorem map_filter {f : α ≃ β} {p : α → Prop} [DecidablePred p] :
    (s.filter p).map f.toEmbedding = (s.map f.toEmbedding).filter (p ∘ f.symm) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    f : Equiv α β
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finset.map f.toEmbedding (Finset.filter p s)) (Finset.filter (Function.c …
  -/
  simp only [filter_map, Function.comp_def, Equiv.toEmbedding_apply, Equiv.symm_apply_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_map {s t : Finset α} (f : α ↪ β) :
    Disjoint (s.map f) (t.map f) ↔ Disjoint s t :=
  mod_cast Set.disjoint_image_iff f.injective (s := s) (t := t)


theorem map_disjUnion {f : α ↪ β} (s₁ s₂ : Finset α) (h) (h' := (disjoint_map _).mpr h) :
    (s₁.disjUnion s₂ h).map f = (s₁.map f).disjUnion (s₂.map f) h' :=
  eq_of_veq <| Multiset.map_add _ _ _


/-- A version of `Finset.map_disjUnion` for writing in the other direction. -/
theorem map_disjUnion' {f : α ↪ β} (s₁ s₂ : Finset α) (h') (h := (disjoint_map _).mp h') :
    (s₁.disjUnion s₂ h).map f = (s₁.map f).disjUnion (s₂.map f) h' :=
  map_disjUnion _ _ _


theorem map_union [DecidableEq α] [DecidableEq β] {f : α ↪ β} (s₁ s₂ : Finset α) :
    (s₁ ∪ s₂).map f = s₁.map f ∪ s₂.map f :=
  mod_cast Set.image_union f s₁ s₂


theorem map_inter [DecidableEq α] [DecidableEq β] {f : α ↪ β} (s₁ s₂ : Finset α) :
    (s₁ ∩ s₂).map f = s₁.map f ∩ s₂.map f :=
  mod_cast Set.image_inter f.injective (s := s₁) (t := s₂)


@[simp]
theorem map_singleton (f : α ↪ β) (a : α) : map f {a} = {f a} :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        f : Function.Embedding α β
                        a : α
                        ⊢ Eq ↑(Finset.map f (Singleton.singleton a)) ↑(Singleton.singleton (f a))
                      -/
  coe_injective <| by simp only [coe_map, coe_singleton, Set.image_singleton]
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem map_insert [DecidableEq α] [DecidableEq β] (f : α ↪ β) (a : α) (s : Finset α) :
    (insert a s).map f = insert (f a) (s.map f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    f : Function.Embedding α β
    a : α
    s : Finset α
    ⊢ Eq (Finset.map f (Insert.insert a s)) (Insert.insert (f a) (Finset.map f s))
  -/
  simp only [insert_eq, map_union, map_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem map_cons (f : α ↪ β) (a : α) (s : Finset α) (ha : a ∉ s) :
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     γ : Type u_3
                                                     f✝ : Function.Embedding α β
                                                     s✝ : Finset α
                                                     f : Function.Embedding α β
                                                     a : α
                                                     s : Finset α
                                                     ha : Not (Membership.mem s a)
                                                     ⊢ Not (Membership.mem (Finset.map f s) (f a))
                                                   -/
    (cons a s ha).map f = cons (f a) (s.map f) (by simpa using ha) :=
                                                   /-
                                                     🎉 no goals
                                                   -/
  eq_of_veq <| Multiset.map_cons f a s.val


@[simp]
theorem map_eq_empty : s.map f = ∅ ↔ s = ∅ := (map_injective f).eq_iff' (map_empty f)


@[simp]
theorem map_nonempty : (s.map f).Nonempty ↔ s.Nonempty :=
  mod_cast Set.image_nonempty (f := f) (s := s)


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected alias ⟨_, Nonempty.map⟩ := map_nonempty


@[simp]
theorem map_nontrivial : (s.map f).Nontrivial ↔ s.Nontrivial :=
  mod_cast Set.image_nontrivial f.injective (s := s)


theorem attach_map_val {s : Finset α} : s.attach.map (Embedding.subtype _) = s :=
                  /-
                    α : Type u_1
                    s : Finset α
                    ⊢ Eq (Finset.map (Function.Embedding.subtype fun x => Membership.mem s x) s.at …
                  -/
  eq_of_veq <| by rw [map_val, attach_val]; exact Multiset.attach_map_val _
                                            /-
                                              🎉 no goals
                                            -/


theorem disjoint_range_addLeftEmbedding (a : ℕ) (s : Finset ℕ) :
    Disjoint (range a) (map (addLeftEmbedding a) s) := by
  /-
    a : Nat
    s : Finset Nat
    ⊢ Disjoint (Finset.range a) (Finset.map (addLeftEmbedding a) s)
  -/
  simp_rw [disjoint_left, mem_map, mem_range, addLeftEmbedding_apply]
  /-
    a : Nat
    s : Finset Nat
    ⊢ ∀ ⦃a_1 : Nat⦄, LT.lt a_1 a → Not (Exists fun a_3 => And (Membership.mem s a_ …
  -/
  rintro _ h ⟨l, -, rfl⟩
  /-
    case intro.intro
    a : Nat
    s : Finset Nat
    l : Nat
    h : LT.lt (HAdd.hAdd a l) a
    ⊢ False
  -/
  omega
  /-
    🎉 no goals
  -/


theorem disjoint_range_addRightEmbedding (a : ℕ) (s : Finset ℕ) :
    Disjoint (range a) (map (addRightEmbedding a) s) := by
  /-
    a : Nat
    s : Finset Nat
    ⊢ Disjoint (Finset.range a) (Finset.map (addRightEmbedding a) s)
  -/
  rw [← addLeftEmbedding_eq_addRightEmbedding]
  /-
    a : Nat
    s : Finset Nat
    ⊢ Disjoint (Finset.range a) (Finset.map (addLeftEmbedding a) s)
  -/
  apply disjoint_range_addLeftEmbedding
  /-
    🎉 no goals
  -/


theorem map_disjiUnion {f : α ↪ β} {s : Finset α} {t : β → Finset γ} {h} :
    (s.map f).disjiUnion t h =
      s.disjiUnion (fun a => t (f a)) fun _ ha _ hb hab =>
        h (mem_map_of_mem _ ha) (mem_map_of_mem _ hb) (f.injective.ne hab) :=
  eq_of_veq <| Multiset.bind_map _ _ _


theorem disjiUnion_map {s : Finset α} {t : α → Finset β} {f : β ↪ γ} {h} :
    (s.disjiUnion t h).map f =
      s.disjiUnion (fun a => (t a).map f) (h.mono' fun _ _ ↦ (disjoint_map _).2) :=
  eq_of_veq <| Multiset.map_bind _ _ _


theorem range_add_one' (n : ℕ) :
                                                                           /-
                                                                             α : Type u_1
                                                                             β : Type u_2
                                                                             γ : Type u_3
                                                                             n i j : Nat
                                                                             ⊢ Eq ((fun i => HAdd.hAdd i 1) i) ((fun i => HAdd.hAdd i 1) j) → Eq i j
                                                                           -/
    range (n + 1) = insert 0 ((range n).map ⟨fun i => i + 1, fun i j => by simp⟩) := by
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  /-
    n : Nat
    ⊢ Eq (Finset.range (HAdd.hAdd n 1)) (Insert.insert 0 (Finset.map { toFun := fu …
  -/
                     /-
                       🎉 no goals
                     -/
  ext (⟨⟩ | ⟨n⟩) <;> simp [Nat.zero_lt_succ n]
                     /-
                       🎉 no goals
                     -/


/-- `image f s` is the forward image of `s` under `f`. -/
def image (f : α → β) (s : Finset α) : Finset β :=
  (s.1.map f).toFinset


@[simp]
theorem image_val (f : α → β) (s : Finset α) : (image f s).1 = (s.1.map f).dedup :=
  rfl


@[simp]
theorem image_empty (f : α → β) : (∅ : Finset α).image f = ∅ :=
  rfl


@[simp]
theorem mem_image : b ∈ s.image f ↔ ∃ a ∈ s, f a = b := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    b : β
    ⊢ Iff (Membership.mem (Finset.image f s) b) (Exists fun a => And (Membership.m …
  -/
  simp only [mem_def, image_val, mem_dedup, Multiset.mem_map, exists_prop]
  /-
    🎉 no goals
  -/


theorem mem_image_of_mem (f : α → β) {a} (h : a ∈ s) : f a ∈ s.image f :=
  mem_image.2 ⟨_, h, rfl⟩


                                                                                              /-
                                                                                                α : Type u_1
                                                                                                β : Type u_2
                                                                                                inst✝ : DecidableEq β
                                                                                                f : α → β
                                                                                                s : Finset α
                                                                                                p : β → Prop
                                                                                                ⊢ Iff (∀ (y : β), Membership.mem (Finset.image f s) y → p y) (∀ ⦃x : α⦄, Membe …
                                                                                              -/
lemma forall_mem_image {p : β → Prop} : (∀ y ∈ s.image f, p y) ↔ ∀ ⦃x⦄, x ∈ s → p (f x) := by simp
                                                                                              /-
                                                                                                🎉 no goals
                                                                                              -/

                                                                                        /-
                                                                                          α : Type u_1
                                                                                          β : Type u_2
                                                                                          inst✝ : DecidableEq β
                                                                                          f : α → β
                                                                                          s : Finset α
                                                                                          p : β → Prop
                                                                                          ⊢ Iff (Exists fun y => And (Membership.mem (Finset.image f s) y) (p y)) (Exist …
                                                                                        -/
lemma exists_mem_image {p : β → Prop} : (∃ y ∈ s.image f, p y) ↔ ∃ x ∈ s, p (f x) := by simp
                                                                                        /-
                                                                                          🎉 no goals
                                                                                        -/


@[deprecated (since := "2024-11-23")] alias forall_image := forall_mem_image


theorem map_eq_image (f : α ↪ β) (s : Finset α) : s.map f = s.image f :=
  eq_of_veq (s.map f).2.dedup.symm

--@[simp] Porting note: removing simp, `simp` [Nonempty] can prove it

theorem mem_image_const : c ∈ s.image (const α b) ↔ s.Nonempty ∧ b = c := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    b c : β
    ⊢ Iff (Membership.mem (Finset.image (Function.const α b) s) c) (And s.Nonempty …
  -/
  rw [mem_image]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    b c : β
    ⊢ Iff (Exists fun a => And (Membership.mem s a) (Eq (Function.const α b a) c)) …
  -/
  simp only [exists_prop, const_apply, exists_and_right]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    b c : β
    ⊢ Iff (And (Exists fun x => Membership.mem s x) (Eq b c)) (And s.Nonempty (Eq  …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_image_const_self : b ∈ s.image (const α b) ↔ s.Nonempty :=
  mem_image_const.trans <| and_iff_left rfl


instance canLift (c) (p) [CanLift β α c p] :
    CanLift (Finset β) (Finset α) (image c) fun s => ∀ x ∈ s, p x where
  prf := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : DecidableEq β
      f g : α → β
      s : Finset α
      t : Finset β
      a : α
      b c✝ : β
      c : α → β
      p : β → Prop
      inst✝ : CanLift β α c p
      ⊢ ∀ (x : Finset β), (∀ (x_1 : β), Membership.mem x x_1 → p x_1) → Exists fun y …
    -/
    rintro ⟨⟨l⟩, hd : l.Nodup⟩ hl
    /-
      case mk.mk
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : DecidableEq β
      f g : α → β
      s : Finset α
      t : Finset β
      a : α
      b c✝ : β
      c : α → β
      p : β → Prop
      inst✝ : CanLift β α c p
      val✝ : Multiset β
      l : List β
      hd : l.Nodup
      hl : ∀ (x : β), Membership.mem { val := Quot.mk (⇑(List.isSetoid β)) l, nodup  …
      ⊢ Exists fun y => Eq (Finset.image c y) { val := Quot.mk (⇑(List.isSetoid β))  …
    -/
    lift l to List α using hl
    /-
      case mk.mk.intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      inst✝¹ : DecidableEq β
      f g : α → β
      s : Finset α
      t : Finset β
      a : α
      b c✝ : β
      c : α → β
      p : β → Prop
      inst✝ : CanLift β α c p
      val✝ : Multiset β
      l : List α
      hd : (List.map c l).Nodup
      ⊢ Exists fun y => Eq (Finset.image c y) { val := Quot.mk (⇑(List.isSetoid β))  …
    -/
    exact ⟨⟨l, hd.of_map _⟩, ext fun a => by simp⟩
    /-
      🎉 no goals
    -/


theorem image_congr (h : (s : Set α).EqOn f g) : Finset.image f s = Finset.image g s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f g : α → β
    s : Finset α
    h : Set.EqOn f g ↑s
    ⊢ Eq (Finset.image f s) (Finset.image g s)
  -/
  ext
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f g : α → β
    s : Finset α
    h : Set.EqOn f g ↑s
    a✝ : β
    ⊢ Iff (Membership.mem (Finset.image f s) a✝) (Membership.mem (Finset.image g s …
  -/
  simp_rw [mem_image, ← bex_def]
  /-
    case h
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f g : α → β
    s : Finset α
    h : Set.EqOn f g ↑s
    a✝ : β
    ⊢ Iff (Exists fun x => Exists fun x_1 => Eq (f x) a✝) (Exists fun x => Exists  …
  -/
  exact exists₂_congr fun x hx => by rw [h hx]
  /-
    🎉 no goals
  -/


theorem _root_.Function.Injective.mem_finset_image (hf : Injective f) :
    f a ∈ s.image f ↔ a ∈ s := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    a : α
    hf : Function.Injective f
    ⊢ Iff (Membership.mem (Finset.image f s) (f a)) (Membership.mem s a)
  -/
  refine ⟨fun h => ?_, Finset.mem_image_of_mem f⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    a : α
    hf : Function.Injective f
    h : Membership.mem (Finset.image f s) (f a)
    ⊢ Membership.mem s a
  -/
  obtain ⟨y, hy, heq⟩ := mem_image.1 h
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    a : α
    hf : Function.Injective f
    h : Membership.mem (Finset.image f s) (f a)
    y : α
    hy : Membership.mem s y
    heq : Eq (f y) (f a)
    ⊢ Membership.mem s a
  -/
  exact hf heq ▸ hy
  /-
    🎉 no goals
  -/



@[simp, norm_cast]
theorem coe_image : ↑(s.image f) = f '' ↑s :=
                /-
                  α : Type u_1
                  β : Type u_2
                  inst✝ : DecidableEq β
                  f : α → β
                  s : Finset α
                  ⊢ ∀ (x : β), Iff (Membership.mem (↑(Finset.image f s)) x) (Membership.mem (Set …
                -/
  Set.ext <| by simp only [mem_coe, mem_image, Set.mem_image, implies_true]
                /-
                  🎉 no goals
                -/


@[simp]
lemma image_nonempty : (s.image f).Nonempty ↔ s.Nonempty :=
  mod_cast Set.image_nonempty (f := f) (s := (s : Set α))


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected theorem Nonempty.image (h : s.Nonempty) (f : α → β) : (s.image f).Nonempty :=
  image_nonempty.2 h


alias ⟨Nonempty.of_image, _⟩ := image_nonempty


theorem image_toFinset [DecidableEq α] {s : Multiset α} :
    s.toFinset.image f = (s.map f).toFinset :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝¹ : DecidableEq β
                    f : α → β
                    inst✝ : DecidableEq α
                    s : Multiset α
                    x✝ : β
                    ⊢ Iff (Membership.mem (Finset.image f s.toFinset) x✝) (Membership.mem (Multise …
                  -/
  ext fun _ => by simp only [mem_image, Multiset.mem_toFinset, exists_prop, Multiset.mem_map]
                  /-
                    🎉 no goals
                  -/


theorem image_val_of_injOn (H : Set.InjOn f s) : (image f s).1 = s.1.map f :=
  (s.2.map_on H).dedup


@[simp]
theorem image_id [DecidableEq α] : s.image id = s :=
                  /-
                    α : Type u_1
                    s : Finset α
                    inst✝ : DecidableEq α
                    x✝ : α
                    ⊢ Iff (Membership.mem (Finset.image id s) x✝) (Membership.mem s x✝)
                  -/
  ext fun _ => by simp only [mem_image, exists_prop, id, exists_eq_right]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem image_id' [DecidableEq α] : (s.image fun x => x) = s :=
  image_id


theorem image_image [DecidableEq γ] {g : β → γ} : (s.image f).image g = s.image (g ∘ f) :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    γ : Type u_3
                    inst✝¹ : DecidableEq β
                    f : α → β
                    s : Finset α
                    inst✝ : DecidableEq γ
                    g : β → γ
                    ⊢ Eq (Finset.image g (Finset.image f s)).val (Finset.image (Function.comp g f) …
                  -/
  eq_of_veq <| by simp only [image_val, dedup_map_dedup_eq, Multiset.map_map]
                  /-
                    🎉 no goals
                  -/


theorem image_comm {β'} [DecidableEq β'] [DecidableEq γ] {f : β → γ} {g : α → β} {f' : α → β'}
    {g' : β' → γ} (h_comm : ∀ a, f (g a) = g' (f' a)) :
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        γ : Type u_3
                                                        inst✝² : DecidableEq β
                                                        s : Finset α
                                                        β' : Type u_4
                                                        inst✝¹ : DecidableEq β'
                                                        inst✝ : DecidableEq γ
                                                        f : β → γ
                                                        g : α → β
                                                        f' : α → β'
                                                        g' : β' → γ
                                                        h_comm : ∀ (a : α), Eq (f (g a)) (g' (f' a))
                                                        ⊢ Eq (Finset.image f (Finset.image g s)) (Finset.image g' (Finset.image f' s))
                                                      -/
    (s.image g).image f = (s.image f').image g' := by simp_rw [image_image, comp_def, h_comm]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem _root_.Function.Semiconj.finset_image [DecidableEq α] {f : α → β} {ga : α → α} {gb : β → β}
    (h : Function.Semiconj f ga gb) : Function.Semiconj (image f) (image ga) (image gb) := fun _ =>
  image_comm h


theorem _root_.Function.Commute.finset_image [DecidableEq α] {f g : α → α}
    (h : Function.Commute f g) : Function.Commute (image f) (image g) :=
  Function.Semiconj.finset_image h


theorem image_subset_image {s₁ s₂ : Finset α} (h : s₁ ⊆ s₂) : s₁.image f ⊆ s₂.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s₁ s₂ : Finset α
    h : HasSubset.Subset s₁ s₂
    ⊢ HasSubset.Subset (Finset.image f s₁) (Finset.image f s₂)
  -/
  simp only [subset_def, image_val, subset_dedup', dedup_subset', Multiset.map_subset_map h]
  /-
    🎉 no goals
  -/


theorem image_subset_iff : s.image f ⊆ t ↔ ∀ x ∈ s, f x ∈ t :=
  calc
                                       /-
                                         α : Type u_1
                                         β : Type u_2
                                         inst✝ : DecidableEq β
                                         f : α → β
                                         s : Finset α
                                         t : Finset β
                                         ⊢ Iff (HasSubset.Subset (Finset.image f s) t) (HasSubset.Subset (Set.image f ↑ …
                                       -/
    s.image f ⊆ t ↔ f '' ↑s ⊆ ↑t := by norm_cast
                                       /-
                                         🎉 no goals
                                       -/
    _ ↔ _ := Set.image_subset_iff


theorem image_mono (f : α → β) : Monotone (Finset.image f) := fun _ _ => image_subset_image


lemma image_injective (hf : Injective f) : Injective (image f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    hf : Function.Injective f
    ⊢ Function.Injective (Finset.image f)
  -/
  simpa only [funext (map_eq_image _)] using map_injective ⟨f, hf⟩
  /-
    🎉 no goals
  -/


lemma image_inj {t : Finset α} (hf : Injective f) : s.image f = t.image f ↔ s = t :=
  (image_injective hf).eq_iff


theorem image_subset_image_iff {t : Finset α} (hf : Injective f) :
    s.image f ⊆ t.image f ↔ s ⊆ t :=
  mod_cast Set.image_subset_image_iff hf (s := s) (t := t)


lemma image_ssubset_image {t : Finset α} (hf : Injective f) : s.image f ⊂ t.image f ↔ s ⊂ t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s t : Finset α
    hf : Function.Injective f
    ⊢ Iff (HasSSubset.SSubset (Finset.image f s) (Finset.image f t)) (HasSSubset.S …
  -/
  simp_rw [← lt_iff_ssubset]
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s t : Finset α
    hf : Function.Injective f
    ⊢ Iff (LT.lt (Finset.image f s) (Finset.image f t)) (LT.lt s t)
  -/
  exact lt_iff_lt_of_le_iff_le' (image_subset_image_iff hf) (image_subset_image_iff hf)
  /-
    🎉 no goals
  -/


theorem coe_image_subset_range : ↑(s.image f) ⊆ Set.range f :=
  calc
    ↑(s.image f) = f '' ↑s := coe_image
    _ ⊆ Set.range f := Set.image_subset_range f ↑s


theorem filter_image {p : β → Prop} [DecidablePred p] :
    (s.image f).filter p = (s.filter fun a ↦ p (f a)).image f :=
  ext fun b => by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      f : α → β
      s : Finset α
      p : β → Prop
      inst✝ : DecidablePred p
      b : β
      ⊢ Iff (Membership.mem (Finset.filter p (Finset.image f s)) b) (Membership.mem  …
    -/
    simp only [mem_filter, mem_image, exists_prop]
    exact
      ⟨by rintro ⟨⟨x, h1, rfl⟩, h2⟩; exact ⟨x, ⟨h1, h2⟩, rfl⟩,
       by rintro ⟨x, ⟨h1, h2⟩, rfl⟩; exact ⟨⟨x, h1, rfl⟩, h2⟩⟩


@[deprecated filter_mem_eq_inter (since := "2024-09-15")]
theorem filter_mem_image_eq_image (f : α → β) (s : Finset α) (t : Finset β) (h : ∀ x ∈ s, f x ∈ t) :
    (t.filter fun y => y ∈ s.image f) = s.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    t : Finset β
    h : ∀ (x : α), Membership.mem s x → Membership.mem t (f x)
    ⊢ Eq (Finset.filter (fun y => Membership.mem (Finset.image f s) y) t) (Finset. …
  -/
  rwa [filter_mem_eq_inter, inter_eq_right, image_subset_iff]
  /-
    🎉 no goals
  -/


theorem fiber_nonempty_iff_mem_image {y : β} : (s.filter (f · = y)).Nonempty ↔ y ∈ s.image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    y : β
    ⊢ Iff (Finset.filter (fun x => Eq (f x) y) s).Nonempty (Membership.mem (Finset …
  -/
  simp [Finset.Nonempty]
  /-
    🎉 no goals
  -/


theorem image_union [DecidableEq α] {f : α → β} (s₁ s₂ : Finset α) :
    (s₁ ∪ s₂).image f = s₁.image f ∪ s₂.image f :=
  mod_cast Set.image_union f s₁ s₂


theorem image_inter_subset [DecidableEq α] (f : α → β) (s t : Finset α) :
    (s ∩ t).image f ⊆ s.image f ∩ t.image f :=
  (image_mono f).map_inf_le s t


theorem image_inter_of_injOn [DecidableEq α] {f : α → β} (s t : Finset α)
    (hf : Set.InjOn f (s ∪ t)) : (s ∩ t).image f = s.image f ∩ t.image f :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      inst✝ : DecidableEq α
      f : α → β
      s t : Finset α
      hf : Set.InjOn f (Union.union ↑s ↑t)
      ⊢ Eq ↑(Finset.image f (Inter.inter s t)) ↑(Inter.inter (Finset.image f s) (Fin …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq β
      inst✝ : DecidableEq α
      f : α → β
      s t : Finset α
      hf : Set.InjOn f (Union.union ↑s ↑t)
      ⊢ Eq (Set.image f (Inter.inter ↑s ↑t)) (Inter.inter (Set.image f ↑s) (Set.imag …
    -/
    exact Set.image_inter_on fun a ha b hb => hf (Or.inr ha) <| Or.inl hb
    /-
      🎉 no goals
    -/


theorem image_inter [DecidableEq α] (s₁ s₂ : Finset α) (hf : Injective f) :
    (s₁ ∩ s₂).image f = s₁.image f ∩ s₂.image f :=
  image_inter_of_injOn _ _ hf.injOn


@[simp]
theorem image_singleton (f : α → β) (a : α) : image f {a} = {f a} :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝ : DecidableEq β
                    f : α → β
                    a : α
                    x : β
                    ⊢ Iff (Membership.mem (Finset.image f (Singleton.singleton a)) x) (Membership. …
                  -/
  ext fun x => by simpa only [mem_image, exists_prop, mem_singleton, exists_eq_left] using eq_comm
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem image_insert [DecidableEq α] (f : α → β) (a : α) (s : Finset α) :
    (insert a s).image f = insert (f a) (s.image f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    f : α → β
    a : α
    s : Finset α
    ⊢ Eq (Finset.image f (Insert.insert a s)) (Insert.insert (f a) (Finset.image f …
  -/
  simp only [insert_eq, image_singleton, image_union]
  /-
    🎉 no goals
  -/


theorem erase_image_subset_image_erase [DecidableEq α] (f : α → β) (s : Finset α) (a : α) :
    (s.image f).erase (f a) ⊆ (s.erase a).image f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    f : α → β
    s : Finset α
    a : α
    ⊢ HasSubset.Subset ((Finset.image f s).erase (f a)) (Finset.image f (s.erase a))
  -/
  simp only [subset_iff, and_imp, exists_prop, mem_image, exists_imp, mem_erase]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    f : α → β
    s : Finset α
    a : α
    ⊢ ∀ ⦃x : β⦄, Ne x (f a) → ∀ (x_1 : α), Membership.mem s x_1 → Eq (f x_1) x → E …
  -/
  rintro b hb x hx rfl
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    f : α → β
    s : Finset α
    a x : α
    hx : Membership.mem s x
    hb : Ne (f x) (f a)
    ⊢ Exists fun a_1 => And (And (Ne a_1 a) (Membership.mem s a_1)) (Eq (f a_1) (f …
  -/
  exact ⟨_, ⟨ne_of_apply_ne f hb, hx⟩, rfl⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem image_erase [DecidableEq α] {f : α → β} (hf : Injective f) (s : Finset α) (a : α) :
    (s.erase a).image f = (s.image f).erase (f a) :=
                      /-
                        α : Type u_1
                        β : Type u_2
                        inst✝¹ : DecidableEq β
                        inst✝ : DecidableEq α
                        f : α → β
                        hf : Function.Injective f
                        s : Finset α
                        a : α
                        ⊢ Eq ↑(Finset.image f (s.erase a)) ↑((Finset.image f s).erase (f a))
                      -/
  coe_injective <| by push_cast [Set.image_diff hf, Set.image_singleton]; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
theorem image_eq_empty : s.image f = ∅ ↔ s = ∅ := mod_cast Set.image_eq_empty (f := f) (s := s)


theorem image_sdiff [DecidableEq α] {f : α → β} (s t : Finset α) (hf : Injective f) :
    (s \ t).image f = s.image f \ t.image f :=
  mod_cast Set.image_diff hf s t


lemma image_sdiff_of_injOn [DecidableEq α] {t : Finset α} (hf : Set.InjOn f s) (hts : t ⊆ s) :
    (s \ t).image f = s.image f \ t.image f :=
  mod_cast Set.image_diff_of_injOn hf <| coe_subset.2 hts


open scoped symmDiff in
theorem image_symmDiff [DecidableEq α] {f : α → β} (s t : Finset α) (hf : Injective f) :
    (s ∆ t).image f = s.image f ∆ t.image f :=
  mod_cast Set.image_symmDiff hf s t


@[simp]
theorem _root_.Disjoint.of_image_finset {s t : Finset α} {f : α → β}
    (h : Disjoint (s.image f) (t.image f)) : Disjoint s t :=
  disjoint_iff_ne.2 fun _ ha _ hb =>
    ne_of_apply_ne f <| h.forall_ne_finset (mem_image_of_mem _ ha) (mem_image_of_mem _ hb)


theorem mem_range_iff_mem_finset_range_of_mod_eq' [DecidableEq α] {f : ℕ → α} {a : α} {n : ℕ}
    (hn : 0 < n) (h : ∀ i, f (i % n) = f i) :
    a ∈ Set.range f ↔ a ∈ (Finset.range n).image fun i => f i := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    f : Nat → α
    a : α
    n : Nat
    hn : LT.lt 0 n
    h : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
    ⊢ Iff (Membership.mem (Set.range f) a) (Membership.mem (Finset.image (fun i => …
  -/
  constructor
    /-
      case mp
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      ⊢ Membership.mem (Set.range f) a → Membership.mem (Finset.image (fun i => f i) …
    -/
  · rintro ⟨i, hi⟩
    /-
      case mp.intro
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      i : Nat
      hi : Eq (f i) a
      ⊢ Membership.mem (Finset.image (fun i => f i) (Finset.range n)) a
    -/
    simp only [mem_image, exists_prop, mem_range]
    /-
      case mp.intro
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      i : Nat
      hi : Eq (f i) a
      ⊢ Exists fun a_1 => And (LT.lt a_1 n) (Eq (f a_1) a)
    -/
    exact ⟨i % n, Nat.mod_lt i hn, (rfl.congr hi).mp (h i)⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      ⊢ Membership.mem (Finset.image (fun i => f i) (Finset.range n)) a → Membership …
    -/
  · rintro h
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h✝ : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      h : Membership.mem (Finset.image (fun i => f i) (Finset.range n)) a
      ⊢ Membership.mem (Set.range f) a
    -/
    simp only [mem_image, exists_prop, Set.mem_range, mem_range] at *
    /-
      case mpr
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h✝ : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      h : Exists fun a_1 => And (LT.lt a_1 n) (Eq (f a_1) a)
      ⊢ Exists fun y => Eq (f y) a
    -/
    rcases h with ⟨i, _, ha⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      f : Nat → α
      a : α
      n : Nat
      hn : LT.lt 0 n
      h : ∀ (i : Nat), Eq (f (HMod.hMod i n)) (f i)
      i : Nat
      left✝ : LT.lt i n
      ha : Eq (f i) a
      ⊢ Exists fun y => Eq (f y) a
    -/
    exact ⟨i, ha⟩
    /-
      🎉 no goals
    -/


theorem mem_range_iff_mem_finset_range_of_mod_eq [DecidableEq α] {f : ℤ → α} {a : α} {n : ℕ}
    (hn : 0 < n) (h : ∀ i, f (i % n) = f i) :
    a ∈ Set.range f ↔ a ∈ (Finset.range n).image (fun (i : ℕ) => f i) :=
                                                           /-
                                                             α : Type u_1
                                                             inst✝ : DecidableEq α
                                                             f : Int → α
                                                             a : α
                                                             n : Nat
                                                             hn : LT.lt 0 n
                                                             h : ∀ (i : Int), Eq (f (HMod.hMod i ↑n)) (f i)
                                                             this : Iff (Exists fun i => Eq (f (HMod.hMod i ↑n)) a) (Exists fun i => And (L …
                                                             ⊢ Iff (Membership.mem (Set.range f) a) (Membership.mem (Finset.image (fun i => …
                                                           -/
  suffices (∃ i, f (i % n) = a) ↔ ∃ i, i < n ∧ f ↑i = a by simpa [h]
                                                           /-
                                                             🎉 no goals
                                                           -/
  have hn' : 0 < (n : ℤ) := Int.ofNat_lt.mpr hn
  Iff.intro
    (fun ⟨i, hi⟩ =>
        /-
          α : Type u_1
          inst✝ : DecidableEq α
          f : Int → α
          a : α
          n : Nat
          hn : LT.lt 0 n
          h : ∀ (i : Int), Eq (f (HMod.hMod i ↑n)) (f i)
          hn' : LT.lt 0 ↑n
          x✝ : Exists fun i => Eq (f (HMod.hMod i ↑n)) a
          i : Int
          hi : Eq (f (HMod.hMod i ↑n)) a
          this : LE.le 0 (HMod.hMod i ↑n)
          ⊢ And (LT.lt (HMod.hMod i ↑n).toNat n) (Eq (f ↑(HMod.hMod i ↑n).toNat) a)
        -/
      have : 0 ≤ i % ↑n := Int.emod_nonneg _ (ne_of_gt hn')
                                                       /-
                                                         🎉 no goals
                                                       -/
      ⟨Int.toNat (i % n), by
           /-
             α : Type u_1
             inst✝ : DecidableEq α
             f : Int → α
             a : α
             n : Nat
             hn : LT.lt 0 n
             h : ∀ (i : Int), Eq (f (HMod.hMod i ↑n)) (f i)
             hn' : LT.lt 0 ↑n
             x✝ : Exists fun i => And (LT.lt i n) (Eq (f ↑i) a)
             i : Nat
             hi : LT.lt i n
             ha : Eq (f ↑i) a
             ⊢ Eq (f (HMod.hMod ↑i ↑n)) a
           -/
        rw [← Int.ofNat_lt, Int.toNat_of_nonneg this]; exact ⟨Int.emod_lt_of_pos i hn', hi⟩⟩)
           /-
             🎉 no goals
           -/
    fun ⟨i, hi, ha⟩ =>
    ⟨i, by rw [Int.emod_eq_of_lt (Int.ofNat_zero_le _) (Int.ofNat_lt_ofNat_of_lt hi), ha]⟩


theorem range_add (a b : ℕ) : range (a + b) = range a ∪ (range b).map (addLeftEmbedding a) := by
  /-
    a b : Nat
    ⊢ Eq (Finset.range (HAdd.hAdd a b)) (Union.union (Finset.range a) (Finset.map  …
  -/
  rw [← val_inj, union_val]
  /-
    a b : Nat
    ⊢ Eq (Finset.range (HAdd.hAdd a b)).val (Union.union (Finset.range a).val (Fin …
  -/
  exact Multiset.range_add_eq_union a b
  /-
    🎉 no goals
  -/


@[simp]
theorem attach_image_val [DecidableEq α] {s : Finset α} : s.attach.image Subtype.val = s :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    s : Finset α
                    ⊢ Eq (Finset.image Subtype.val s.attach).val s.val
                  -/
  eq_of_veq <| by rw [image_val, attach_val, Multiset.attach_map_val, dedup_eq_self]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem attach_insert [DecidableEq α] {a : α} {s : Finset α} :
    attach (insert a s) =
      insert (⟨a, mem_insert_self a s⟩ : { x // x ∈ insert a s })
        ((attach s).image fun x => ⟨x.1, mem_insert_of_mem x.2⟩) :=
  ext fun ⟨x, hx⟩ =>
    ⟨Or.casesOn (mem_insert.1 hx)
        (fun h : x = a => fun _ => mem_insert.2 <| Or.inl <| Subtype.eq h) fun h : x ∈ s => fun _ =>
        mem_insert_of_mem <| mem_image.2 <| ⟨⟨x, h⟩, mem_attach _ _, Subtype.eq rfl⟩,
      fun _ => Finset.mem_attach _ _⟩


@[simp]
theorem disjoint_image {s t : Finset α} {f : α → β} (hf : Injective f) :
    Disjoint (s.image f) (t.image f) ↔ Disjoint s t :=
  mod_cast Set.disjoint_image_iff hf (s := s) (t := t)


theorem image_const {s : Finset α} (h : s.Nonempty) (b : β) : (s.image fun _ => b) = singleton b :=
  mod_cast Set.Nonempty.image_const (coe_nonempty.2 h) b


@[simp]
theorem map_erase [DecidableEq α] (f : α ↪ β) (s : Finset α) (a : α) :
    (s.erase a).map f = (s.map f).erase (f a) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    f : Function.Embedding α β
    s : Finset α
    a : α
    ⊢ Eq (Finset.map f (s.erase a)) ((Finset.map f s).erase (f a))
  -/
  simp_rw [map_eq_image]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq β
    inst✝ : DecidableEq α
    f : Function.Embedding α β
    s : Finset α
    a : α
    ⊢ Eq (Finset.image (⇑f) (s.erase a)) ((Finset.image (⇑f) s).erase (f a))
  -/
  exact s.image_erase f.2 a
  /-
    🎉 no goals
  -/


theorem image_biUnion [DecidableEq γ] {f : α → β} {s : Finset α} {t : β → Finset γ} :
    (s.image f).biUnion t = s.biUnion fun a => t (f a) :=
  haveI := Classical.decEq α
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 γ : Type u_3
                                                 inst✝¹ : DecidableEq β
                                                 inst✝ : DecidableEq γ
                                                 f : α → β
                                                 s✝ : Finset α
                                                 t : β → Finset γ
                                                 this : DecidableEq α
                                                 a : α
                                                 s : Finset α
                                                 x✝ : Not (Membership.mem s a)
                                                 ih : Eq ((Finset.image f s).biUnion t) (s.biUnion fun a => t (f a))
                                                 ⊢ Eq ((Finset.image f (Insert.insert a s)).biUnion t) ((Insert.insert a s).biU …
                                               -/
  Finset.induction_on s rfl fun a s _ ih => by simp only [image_insert, biUnion_insert, ih]
                                               /-
                                                 🎉 no goals
                                               -/


theorem biUnion_image [DecidableEq γ] {s : Finset α} {t : α → Finset β} {f : β → γ} :
    (s.biUnion t).image f = s.biUnion fun a => (t a).image f :=
  haveI := Classical.decEq α
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 γ : Type u_3
                                                 inst✝¹ : DecidableEq β
                                                 inst✝ : DecidableEq γ
                                                 s✝ : Finset α
                                                 t : α → Finset β
                                                 f : β → γ
                                                 this : DecidableEq α
                                                 a : α
                                                 s : Finset α
                                                 x✝ : Not (Membership.mem s a)
                                                 ih : Eq (Finset.image f (s.biUnion t)) (s.biUnion fun a => Finset.image f (t a))
                                                 ⊢ Eq (Finset.image f ((Insert.insert a s).biUnion t)) ((Insert.insert a s).biU …
                                               -/
  Finset.induction_on s rfl fun a s _ ih => by simp only [biUnion_insert, image_union, ih]
                                               /-
                                                 🎉 no goals
                                               -/


theorem image_biUnion_filter_eq [DecidableEq α] (s : Finset β) (g : β → α) :
    ((s.image g).biUnion fun a => s.filter fun c => g c = a) = s :=
  biUnion_filter_eq_of_maps_to fun _ => mem_image_of_mem g


theorem biUnion_singleton {f : α → β} : (s.biUnion fun a => {f a}) = s.image f :=
                  /-
                    α : Type u_1
                    β : Type u_2
                    inst✝ : DecidableEq β
                    s : Finset α
                    f : α → β
                    x : β
                    ⊢ Iff (Membership.mem (s.biUnion fun a => Singleton.singleton (f a)) x) (Membe …
                  -/
  ext fun x => by simp only [mem_biUnion, mem_image, mem_singleton, eq_comm]
                  /-
                    🎉 no goals
                  -/


/-- `filterMap f s` is a combination filter/map operation on `s`.
  The function `f : α → Option β` is applied to each element of `s`;
  if `f a` is `some b` then `b` is included in the result, otherwise
  `a` is excluded from the resulting finset.

  In notation, `filterMap f s` is the finset `{b : β | ∃ a ∈ s , f a = some b}`. -/
-- TODO: should there be `filterImage` too?
def filterMap (f : α → Option β) (s : Finset α)
    (f_inj : ∀ a a' b, b ∈ f a → b ∈ f a' → a = a') : Finset β :=
  ⟨s.val.filterMap f, s.nodup.filterMap f f_inj⟩


@[simp]
theorem filterMap_val : (filterMap f s' f_inj).1 = s'.1.filterMap f := rfl


@[simp]
theorem filterMap_empty : (∅ : Finset α).filterMap f f_inj = ∅ := rfl


@[simp]
theorem mem_filterMap {b : β} : b ∈ s.filterMap f f_inj ↔ ∃ a ∈ s, f a = some b :=
  s.val.mem_filterMap f


@[simp, norm_cast]
theorem coe_filterMap : (s.filterMap f f_inj : Set β) = {b | ∃ a ∈ s, f a = some b} :=
              /-
                α : Type u_1
                β : Type u_2
                f : α → Option β
                s : Finset α
                f_inj : ∀ (a a' : α) (b : β), Membership.mem (f a) b → Membership.mem (f a') b …
                ⊢ ∀ (x : β), Iff (Membership.mem (↑(Finset.filterMap f s f_inj)) x) (Membershi …
              -/
  Set.ext (by simp only [mem_coe, mem_filterMap, Option.mem_def, Set.mem_setOf_eq, implies_true])
              /-
                🎉 no goals
              -/


@[simp]
                                              /-
                                                α : Type u_1
                                                β : Type u_2
                                                γ : Type u_3
                                                f : α → Option β
                                                s' s t : Finset α
                                                f_inj : ∀ (a a' : α) (b : β), Membership.mem (f a) b → Membership.mem (f a') b …
                                                ⊢ ∀ (a a' b : α), Membership.mem (Option.some a) b → Membership.mem (Option.so …
                                              -/
theorem filterMap_some : s.filterMap some (by simp) = s :=
                                              /-
                                                🎉 no goals
                                              -/
                  /-
                    α : Type u_1
                    s : Finset α
                    x✝ : α
                    ⊢ Iff (Membership.mem (Finset.filterMap Option.some s ⋯) x✝) (Membership.mem s …
                  -/
  ext fun _ => by simp only [mem_filterMap, Option.some.injEq, exists_eq_right]
                  /-
                    🎉 no goals
                  -/


theorem filterMap_mono (h : s ⊆ t) :
    filterMap f s f_inj ⊆ filterMap f t f_inj := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → Option β
    s t : Finset α
    f_inj : ∀ (a a' : α) (b : β), Membership.mem (f a) b → Membership.mem (f a') b …
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset (Finset.filterMap f s f_inj) (Finset.filterMap f t f_inj)
  -/
  rw [← val_le_iff] at h ⊢
  /-
    α : Type u_1
    β : Type u_2
    f : α → Option β
    s t : Finset α
    f_inj : ∀ (a a' : α) (b : β), Membership.mem (f a) b → Membership.mem (f a') b …
    h : LE.le s.val t.val
    ⊢ LE.le (Finset.filterMap f s f_inj).val (Finset.filterMap f t f_inj).val
  -/
  exact Multiset.filterMap_le_filterMap f h
  /-
    🎉 no goals
  -/


/-- Given a finset `s` and a predicate `p`, `s.subtype p` is the finset of `Subtype p` whose
elements belong to `s`. -/
protected def subtype {α} (p : α → Prop) [DecidablePred p] (s : Finset α) : Finset (Subtype p) :=
  (s.filter p).attach.map
                       /-
                         α✝ : Type u_1
                         β : Type u_2
                         γ : Type u_3
                         α : Type ?u.79287
                         p : α → Prop
                         inst✝ : DecidablePred p
                         s : Finset α
                         x : Subtype fun x => Membership.mem (Finset.filter p s) x
                         ⊢ p ↑x
                       -/
    ⟨fun x => ⟨x.1, by simpa using (Finset.mem_filter.1 x.2).2⟩,
                       /-
                         🎉 no goals
                       -/
     fun _ _ H => Subtype.eq <| Subtype.mk.inj H⟩


@[simp]
theorem mem_subtype {p : α → Prop} [DecidablePred p] {s : Finset α} :
    ∀ {a : Subtype p}, a ∈ s.subtype p ↔ (a : α) ∈ s
                  /-
                    α : Type u_1
                    p : α → Prop
                    inst✝ : DecidablePred p
                    s : Finset α
                    a : α
                    ha : p a
                    ⊢ Iff (Membership.mem (Finset.subtype p s) ⟨a, ha⟩) (Membership.mem s ↑⟨a, ha⟩)
                  -/
  | ⟨a, ha⟩ => by simp [Finset.subtype, ha]
                  /-
                    🎉 no goals
                  -/


theorem subtype_eq_empty {p : α → Prop} [DecidablePred p] {s : Finset α} :
                                             /-
                                               α : Type u_1
                                               p : α → Prop
                                               inst✝ : DecidablePred p
                                               s : Finset α
                                               ⊢ Iff (Eq (Finset.subtype p s) EmptyCollection.emptyCollection) (∀ (x : α), p  …
                                             -/
    s.subtype p = ∅ ↔ ∀ x, p x → x ∉ s := by simp [Finset.ext_iff, Subtype.forall, Subtype.coe_mk]
                                             /-
                                               🎉 no goals
                                             -/


@[mono]
theorem subtype_mono {p : α → Prop} [DecidablePred p] : Monotone (Finset.subtype p) :=
  fun _ _ h _ hx => mem_subtype.2 <| h <| mem_subtype.1 hx


/-- `s.subtype p` converts back to `s.filter p` with
`Embedding.subtype`. -/
@[simp]
theorem subtype_map (p : α → Prop) [DecidablePred p] {s : Finset α} :
    (s.subtype p).map (Embedding.subtype _) = s.filter p := by
  /-
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s : Finset α
    ⊢ Eq (Finset.map (Function.Embedding.subtype p) (Finset.subtype p s)) (Finset. …
  -/
  ext x
  /-
    case h
    α : Type u_1
    p : α → Prop
    inst✝ : DecidablePred p
    s : Finset α
    x : α
    ⊢ Iff (Membership.mem (Finset.map (Function.Embedding.subtype p) (Finset.subty …
  -/
  simp [@and_comm _ (_ = _), @and_left_comm _ (_ = _), @and_comm (p x) (x ∈ s)]
  /-
    🎉 no goals
  -/


/-- If all elements of a `Finset` satisfy the predicate `p`,
`s.subtype p` converts back to `s` with `Embedding.subtype`. -/
theorem subtype_map_of_mem {p : α → Prop} [DecidablePred p] {s : Finset α} (h : ∀ x ∈ s, p x) :
                                                             /-
                                                               α : Type u_1
                                                               p : α → Prop
                                                               inst✝ : DecidablePred p
                                                               s : Finset α
                                                               h : ∀ (x : α), Membership.mem s x → p x
                                                               ⊢ ∀ (a : α), Iff (Membership.mem (Finset.map (Function.Embedding.subtype p) (F …
                                                             -/
    (s.subtype p).map (Embedding.subtype _) = s := ext <| by simpa [subtype_map] using h
                                                             /-
                                                               🎉 no goals
                                                             -/


/-- If a `Finset` of a subtype is converted to the main type with
`Embedding.subtype`, all elements of the result have the property of
the subtype. -/
theorem property_of_mem_map_subtype {p : α → Prop} (s : Finset { x // p x }) {a : α}
    (h : a ∈ s.map (Embedding.subtype _)) : p a := by
  /-
    α : Type u_1
    p : α → Prop
    s : Finset (Subtype fun x => p x)
    a : α
    h : Membership.mem (Finset.map (Function.Embedding.subtype fun x => p x) s) a
    ⊢ p a
  -/
  rcases mem_map.1 h with ⟨x, _, rfl⟩
  /-
    case intro.intro
    α : Type u_1
    p : α → Prop
    s : Finset (Subtype fun x => p x)
    x : Subtype fun x => p x
    left✝ : Membership.mem s x
    h : Membership.mem (Finset.map (Function.Embedding.subtype fun x => p x) s) (( …
    ⊢ p ((Function.Embedding.subtype fun x => p x) x)
  -/
  exact x.2
  /-
    🎉 no goals
  -/


/-- If a `Finset` of a subtype is converted to the main type with
`Embedding.subtype`, the result does not contain any value that does
not satisfy the property of the subtype. -/
theorem not_mem_map_subtype_of_not_property {p : α → Prop} (s : Finset { x // p x }) {a : α}
    (h : ¬p a) : a ∉ s.map (Embedding.subtype _) :=
  mt s.property_of_mem_map_subtype h


/-- If a `Finset` of a subtype is converted to the main type with
`Embedding.subtype`, the result is a subset of the set giving the
subtype. -/
theorem map_subtype_subset {t : Set α} (s : Finset t) : ↑(s.map (Embedding.subtype _)) ⊆ t := by
  /-
    α : Type u_1
    t : Set α
    s : Finset ↑t
    ⊢ HasSubset.Subset (↑(Finset.map (Function.Embedding.subtype fun x => Membersh …
  -/
  intro a ha
  /-
    α : Type u_1
    t : Set α
    s : Finset ↑t
    a : α
    ha : Membership.mem (↑(Finset.map (Function.Embedding.subtype fun x => Members …
    ⊢ Membership.mem t a
  -/
  rw [mem_coe] at ha
  /-
    α : Type u_1
    t : Set α
    s : Finset ↑t
    a : α
    ha : Membership.mem (Finset.map (Function.Embedding.subtype fun x => Membershi …
    ⊢ Membership.mem t a
  -/
  convert property_of_mem_map_subtype s ha
  /-
    🎉 no goals
  -/


/-- Given a finset `s` of natural numbers and a bound `n`,
`s.fin n` is the finset of all elements of `s` less than `n`.
-/
protected def fin (n : ℕ) (s : Finset ℕ) : Finset (Fin n) :=
  (s.subtype _).map Fin.equivSubtype.symm.toEmbedding


@[simp]
theorem mem_fin {n} {s : Finset ℕ} : ∀ a : Fin n, a ∈ s.fin n ↔ (a : ℕ) ∈ s
                  /-
                    n : Nat
                    s : Finset Nat
                    a : Nat
                    ha : LT.lt a n
                    ⊢ Iff (Membership.mem (Finset.fin n s) ⟨a, ha⟩) (Membership.mem s ↑⟨a, ha⟩)
                  -/
  | ⟨a, ha⟩ => by simp [Finset.fin, ha, and_comm]
                  /-
                    🎉 no goals
                  -/


@[mono]
                                                                    /-
                                                                      n : Nat
                                                                      s t : Finset Nat
                                                                      h : LE.le s t
                                                                      x : Fin n
                                                                      ⊢ Membership.mem (Finset.fin n s) x → Membership.mem (Finset.fin n t) x
                                                                    -/
theorem fin_mono {n} : Monotone (Finset.fin n) := fun s t h x => by simpa using @h x
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem fin_map {n} {s : Finset ℕ} : (s.fin n).map Fin.valEmbedding = s.filter (· < n) := by
  /-
    n : Nat
    s : Finset Nat
    ⊢ Eq (Finset.map Fin.valEmbedding (Finset.fin n s)) (Finset.filter (fun x => L …
  -/
  simp [Finset.fin, Finset.map_map]
  /-
    🎉 no goals
  -/


/--
If a finset `t` is a subset of the image of another finset `s` under `f`, then it is equal to the
image of a subset of `s`.

For the version where `s` is a set, see `subset_set_image_iff`.
-/
theorem subset_image_iff [DecidableEq β] {s : Finset α} {t : Finset β} {f : α → β} :
    t ⊆ s.image f ↔ ∃ s' : Finset α, s' ⊆ s ∧ s'.image f = t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    ⊢ Iff (HasSubset.Subset t (Finset.image f s)) (Exists fun s' => And (HasSubset …
  -/
  refine ⟨fun ht => ?_, fun ⟨s', hs', h⟩ => h ▸ image_subset_image hs'⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    ht : HasSubset.Subset t (Finset.image f s)
    ⊢ Exists fun s' => And (HasSubset.Subset s' s) (Eq (Finset.image f s') t)
  -/
  refine ⟨s.filter (f · ∈ t), filter_subset _ _, le_antisymm (by simp [image_subset_iff]) ?_⟩
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    ht : HasSubset.Subset t (Finset.image f s)
    ⊢ LE.le t (Finset.image f (Finset.filter (fun x => Membership.mem t (f x)) s))
  -/
  intro x hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    ht : HasSubset.Subset t (Finset.image f s)
    x : β
    hx : Membership.mem t x
    ⊢ Membership.mem (Finset.image f (Finset.filter (fun x => Membership.mem t (f  …
  -/
  specialize ht hx
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Finset α
    t : Finset β
    f : α → β
    x : β
    hx : Membership.mem t x
    ht : Membership.mem (Finset.image f s) x
    ⊢ Membership.mem (Finset.image f (Finset.filter (fun x => Membership.mem t (f  …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- If a `Finset` is a subset of the image of a `Set` under `f`,
then it is equal to the `Finset.image` of a `Finset` subset of that `Set`. -/
theorem subset_set_image_iff [DecidableEq β] {s : Set α} {t : Finset β} {f : α → β} :
    ↑t ⊆ f '' s ↔ ∃ s' : Finset α, ↑s' ⊆ s ∧ s'.image f = t := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Set α
    t : Finset β
    f : α → β
    ⊢ Iff (HasSubset.Subset (↑t) (Set.image f s)) (Exists fun s' => And (HasSubset …
  -/
  constructor; swap
    /-
      case mpr
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq β
      s : Set α
      t : Finset β
      f : α → β
      ⊢ (Exists fun s' => And (HasSubset.Subset (↑s') s) (Eq (Finset.image f s') t)) …
    -/
  · rintro ⟨t, ht, rfl⟩
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq β
      s : Set α
      f : α → β
      t : Finset α
      ht : HasSubset.Subset (↑t) s
      ⊢ HasSubset.Subset (↑(Finset.image f t)) (Set.image f s)
    -/
    rw [coe_image]
    /-
      case mpr.intro.intro
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq β
      s : Set α
      f : α → β
      t : Finset α
      ht : HasSubset.Subset (↑t) s
      ⊢ HasSubset.Subset (Set.image f ↑t) (Set.image f s)
    -/
    exact Set.image_subset f ht
    /-
      🎉 no goals
    -/
  /-
    case mp
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Set α
    t : Finset β
    f : α → β
    ⊢ HasSubset.Subset (↑t) (Set.image f s) → Exists fun s' => And (HasSubset.Subs …
  -/
  intro h
  /-
    case mp
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Set α
    t : Finset β
    f : α → β
    h : HasSubset.Subset (↑t) (Set.image f s)
    ⊢ Exists fun s' => And (HasSubset.Subset (↑s') s) (Eq (Finset.image f s') t)
  -/
  letI : CanLift β s (f ∘ (↑)) fun y => y ∈ f '' s := ⟨fun y ⟨x, hxt, hy⟩ => ⟨⟨x, hxt⟩, hy⟩⟩
  /-
    case mp
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Set α
    t : Finset β
    f : α → β
    h : HasSubset.Subset (↑t) (Set.image f s)
    this : CanLift β (↑s) (Function.comp f Subtype.val) fun y => Membership.mem (S …
    ⊢ Exists fun s' => And (HasSubset.Subset (↑s') s) (Eq (Finset.image f s') t)
  -/
  lift t to Finset s using h
  /-
    case mp.intro
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Set α
    f : α → β
    this : CanLift β (↑s) (Function.comp f Subtype.val) fun y => Membership.mem (S …
    t : Finset ↑s
    ⊢ Exists fun s' => And (HasSubset.Subset (↑s') s) (Eq (Finset.image f s') (Fin …
  -/
  refine ⟨t.map (Embedding.subtype _), map_subtype_subset _, ?_⟩
  /-
    case mp.intro
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    s : Set α
    f : α → β
    this : CanLift β (↑s) (Function.comp f Subtype.val) fun y => Membership.mem (S …
    t : Finset ↑s
    ⊢ Eq (Finset.image f (Finset.map (Function.Embedding.subtype fun x => Membersh …
  -/
  ext y; simp
         /-
           🎉 no goals
         -/


theorem range_sdiff_zero {n : ℕ} : range (n + 1) \ {0} = (range n).image Nat.succ := by
  /-
    n : Nat
    ⊢ Eq (SDiff.sdiff (Finset.range (HAdd.hAdd n 1)) (Singleton.singleton 0)) (Fin …
  -/
  induction' n with k hk
    /-
      case zero
      ⊢ Eq (SDiff.sdiff (Finset.range (HAdd.hAdd 0 1)) (Singleton.singleton 0)) (Fin …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case succ
    k : Nat
    hk : Eq (SDiff.sdiff (Finset.range (HAdd.hAdd k 1)) (Singleton.singleton 0)) ( …
    ⊢ Eq (SDiff.sdiff (Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)) (Singleton.sing …
  -/
  conv_rhs => rw [range_succ]
  /-
    case succ
    k : Nat
    hk : Eq (SDiff.sdiff (Finset.range (HAdd.hAdd k 1)) (Singleton.singleton 0)) ( …
    ⊢ Eq (SDiff.sdiff (Finset.range (HAdd.hAdd (HAdd.hAdd k 1) 1)) (Singleton.sing …
  -/
  rw [range_succ, image_insert, ← hk, insert_sdiff_of_not_mem]
  /-
    case succ.h
    k : Nat
    hk : Eq (SDiff.sdiff (Finset.range (HAdd.hAdd k 1)) (Singleton.singleton 0)) ( …
    ⊢ Not (Membership.mem (Singleton.singleton 0) (HAdd.hAdd k 1))
  -/
  simp
  /-
    🎉 no goals
  -/


theorem Multiset.toFinset_map [DecidableEq α] [DecidableEq β] (f : α → β) (m : Multiset α) :
    (m.map f).toFinset = m.toFinset.image f :=
  Finset.val_inj.1 (Multiset.dedup_map_dedup_eq _ _).symm


/-- Given an equivalence `α` to `β`, produce an equivalence between `Finset α` and `Finset β`. -/
protected def finsetCongr (e : α ≃ β) : Finset α ≃ Finset β where
  toFun s := s.map e.toEmbedding
  invFun s := s.map e.symm.toEmbedding
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     e : Equiv α β
                     s : Finset α
                     ⊢ Eq ((fun s => Finset.map e.symm.toEmbedding s) ((fun s => Finset.map e.toEmb …
                   -/
  left_inv s := by simp [Finset.map_map]
                   /-
                     🎉 no goals
                   -/
                    /-
                      α : Type u_1
                      β : Type u_2
                      γ : Type u_3
                      e : Equiv α β
                      s : Finset β
                      ⊢ Eq ((fun s => Finset.map e.toEmbedding s) ((fun s => Finset.map e.symm.toEmb …
                    -/
  right_inv s := by simp [Finset.map_map]
                    /-
                      🎉 no goals
                    -/


@[simp]
theorem finsetCongr_apply (e : α ≃ β) (s : Finset α) : e.finsetCongr s = s.map e.toEmbedding :=
  rfl


@[simp]
theorem finsetCongr_refl : (Equiv.refl α).finsetCongr = Equiv.refl _ := by
  /-
    α : Type u_1
    ⊢ Eq (Equiv.refl α).finsetCongr (Equiv.refl (Finset α))
  -/
  ext
  /-
    case H.h
    α : Type u_1
    x✝ : Finset α
    a✝ : α
    ⊢ Iff (Membership.mem ((Equiv.refl α).finsetCongr x✝) a✝) (Membership.mem ((Eq …
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
theorem finsetCongr_symm (e : α ≃ β) : e.finsetCongr.symm = e.symm.finsetCongr :=
  rfl


@[simp]
theorem finsetCongr_trans (e : α ≃ β) (e' : β ≃ γ) :
    e.finsetCongr.trans e'.finsetCongr = (e.trans e').finsetCongr := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    e : Equiv α β
    e' : Equiv β γ
    ⊢ Eq (e.finsetCongr.trans e'.finsetCongr) (e.trans e').finsetCongr
  -/
  ext
  /-
    case H.h
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    e : Equiv α β
    e' : Equiv β γ
    x✝ : Finset α
    a✝ : γ
    ⊢ Iff (Membership.mem ((e.finsetCongr.trans e'.finsetCongr) x✝) a✝) (Membershi …
  -/
  simp [-Finset.mem_map, -Equiv.trans_toEmbedding]
  /-
    🎉 no goals
  -/


theorem finsetCongr_toEmbedding (e : α ≃ β) :
    e.finsetCongr.toEmbedding = (Finset.mapEmbedding e.toEmbedding).toEmbedding :=
  rfl


