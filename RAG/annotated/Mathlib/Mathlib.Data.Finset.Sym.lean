/-- `s.sym2` is the finset of all unordered pairs of elements from `s`.
It is the image of `s ×ˢ s` under the quotient `α × α → Sym2 α`. -/
@[simps]
protected def sym2 (s : Finset α) : Finset (Sym2 α) := ⟨s.1.sym2, s.2.sym2⟩


theorem mk_mem_sym2_iff : s(a, b) ∈ s.sym2 ↔ a ∈ s ∧ b ∈ s := by
  /-
    α : Type u_1
    s : Finset α
    a b : α
    ⊢ Iff (Membership.mem s.sym2 (Sym2.mk { fst := a, snd := b })) (And (Membershi …
  -/
  rw [mem_mk, sym2_val, Multiset.mk_mem_sym2_iff, mem_mk, mem_mk]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_sym2_iff {m : Sym2 α} : m ∈ s.sym2 ↔ ∀ a ∈ m, a ∈ s := by
  /-
    α : Type u_1
    s : Finset α
    m : Sym2 α
    ⊢ Iff (Membership.mem s.sym2 m) (∀ (a : α), Membership.mem m a → Membership.me …
  -/
  rw [mem_mk, sym2_val, Multiset.mem_sym2_iff]
  /-
    α : Type u_1
    s : Finset α
    m : Sym2 α
    ⊢ Iff (∀ (y : α), Membership.mem m y → Membership.mem s.val y) (∀ (a : α), Mem …
  -/
  simp only [mem_val]
  /-
    🎉 no goals
  -/


theorem sym2_cons (a : α) (s : Finset α) (ha : a ∉ s) :
    (s.cons a ha).sym2 = ((s.cons a ha).map <| Sym2.mkEmbedding a).disjUnion s.sym2 (by
      /-
        α : Type u_1
        β : Type u_2
        s✝ t : Finset α
        a✝ b a : α
        s : Finset α
        ha : Not (Membership.mem s a)
        ⊢ Disjoint (Finset.map (Sym2.mkEmbedding a) (Finset.cons a s ha)) s.sym2
      -/
      simp [Finset.disjoint_left, ha]) :=
      /-
        🎉 no goals
      -/
  val_injective <| Multiset.sym2_cons _ _


theorem sym2_insert [DecidableEq α] (a : α) (s : Finset α) :
    (insert a s).sym2 = ((insert a s).image fun b => s(a, b)) ∪ s.sym2 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    s : Finset α
    ⊢ Eq (Insert.insert a s).sym2 (Union.union (Finset.image (fun b => Sym2.mk { f …
  -/
  obtain ha | ha := Decidable.em (a ∈ s)
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      ha : Membership.mem s a
      ⊢ Eq (Insert.insert a s).sym2 (Union.union (Finset.image (fun b => Sym2.mk { f …
    -/
  · simp only [insert_eq_of_mem ha, right_eq_union, image_subset_iff]
    /-
      case inl
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      ha : Membership.mem s a
      ⊢ ∀ (x : α), Membership.mem s x → Membership.mem s.sym2 (Sym2.mk { fst := a, s …
    -/
    aesop
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      inst✝ : DecidableEq α
      a : α
      s : Finset α
      ha : Not (Membership.mem s a)
      ⊢ Eq (Insert.insert a s).sym2 (Union.union (Finset.image (fun b => Sym2.mk { f …
    -/
  · simpa [map_eq_image] using sym2_cons a s ha
    /-
      🎉 no goals
    -/


theorem sym2_map (f : α ↪ β) (s : Finset α) : (s.map f).sym2 = s.sym2.map (.sym2Map f) :=
  val_injective <| s.val.sym2_map _


theorem sym2_image [DecidableEq β] (f : α → β) (s : Finset α) :
    (s.image f).sym2 = s.sym2.image (Sym2.map f) := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    ⊢ Eq (Finset.image f s).sym2 (Finset.image (Sym2.map f) s.sym2)
  -/
  apply val_injective
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    ⊢ Eq (Finset.image f s).sym2.val (Finset.image (Sym2.map f) s.sym2).val
  -/
  dsimp [Finset.sym2]
  /-
    case a
    α : Type u_1
    β : Type u_2
    inst✝ : DecidableEq β
    f : α → β
    s : Finset α
    ⊢ Eq (Multiset.map f s.val).dedup.sym2 (Multiset.map (Sym2.map f) s.val.sym2). …
  -/
  rw [← Multiset.dedup_sym2, Multiset.sym2_map]
  /-
    🎉 no goals
  -/


instance _root_.Sym2.instFintype [Fintype α] : Fintype (Sym2 α) where
  elems := Finset.univ.sym2
                         /-
                           α : Type u_1
                           β : Type u_2
                           s t : Finset α
                           a b : α
                           inst✝ : Fintype α
                           x : Sym2 α
                           ⊢ Membership.mem Finset.univ.sym2 x
                         -/
  complete := fun x ↦ by rw [mem_sym2_iff]; exact (fun a _ ↦ mem_univ a)
                                            /-
                                              🎉 no goals
                                            -/

-- Note(kmill): Using a default argument to make this simp lemma more general.

@[simp]
theorem sym2_univ [Fintype α] (inst : Fintype (Sym2 α) := Sym2.instFintype) :
    (univ : Finset α).sym2 = univ := by
  /-
    α : Type u_1
    inst✝ : Fintype α
    inst : optParam (Fintype (Sym2 α)) Sym2.instFintype
    ⊢ Eq Finset.univ.sym2 Finset.univ
  -/
  ext
  /-
    case h
    α : Type u_1
    inst✝ : Fintype α
    inst : optParam (Fintype (Sym2 α)) Sym2.instFintype
    a✝ : Sym2 α
    ⊢ Iff (Membership.mem Finset.univ.sym2 a✝) (Membership.mem Finset.univ a✝)
  -/
  simp only [mem_sym2_iff, mem_univ, implies_true]
  /-
    🎉 no goals
  -/


@[simp, mono]
theorem sym2_mono (h : s ⊆ t) : s.sym2 ⊆ t.sym2 := by
  /-
    α : Type u_1
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ HasSubset.Subset s.sym2 t.sym2
  -/
  rw [← val_le_iff, sym2_val, sym2_val]
  /-
    α : Type u_1
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ LE.le s.val.sym2 t.val.sym2
  -/
  apply Multiset.sym2_mono
  /-
    case h
    α : Type u_1
    s t : Finset α
    h : HasSubset.Subset s t
    ⊢ LE.le s.val t.val
  -/
  rwa [val_le_iff]
  /-
    🎉 no goals
  -/


theorem monotone_sym2 : Monotone (Finset.sym2 : Finset α → _) := fun _ _ => sym2_mono


theorem injective_sym2 : Function.Injective (Finset.sym2 : Finset α → _) := by
  /-
    α : Type u_1
    ⊢ Function.Injective Finset.sym2
  -/
  intro s t h
  /-
    α : Type u_1
    s t : Finset α
    h : Eq s.sym2 t.sym2
    ⊢ Eq s t
  -/
  ext x
  /-
    case h
    α : Type u_1
    s t : Finset α
    h : Eq s.sym2 t.sym2
    x : α
    ⊢ Iff (Membership.mem s x) (Membership.mem t x)
  -/
  simpa using congr(s(x, x) ∈ $h)
  /-
    🎉 no goals
  -/


theorem strictMono_sym2 : StrictMono (Finset.sym2 : Finset α → _) :=
  monotone_sym2.strictMono_of_injective injective_sym2


theorem sym2_toFinset [DecidableEq α] (m : Multiset α) :
    m.toFinset.sym2 = m.sym2.toFinset := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    ⊢ Eq m.toFinset.sym2 m.sym2.toFinset
  -/
  ext z
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    z : Sym2 α
    ⊢ Iff (Membership.mem m.toFinset.sym2 z) (Membership.mem m.sym2.toFinset z)
  -/
  refine z.ind fun x y ↦ ?_
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    m : Multiset α
    z : Sym2 α
    x y : α
    ⊢ Iff (Membership.mem m.toFinset.sym2 (Sym2.mk { fst := x, snd := y })) (Membe …
  -/
  simp only [mk_mem_sym2_iff, Multiset.mem_toFinset, Multiset.mk_mem_sym2_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem sym2_empty : (∅ : Finset α).sym2 = ∅ := rfl


@[simp]
theorem sym2_eq_empty : s.sym2 = ∅ ↔ s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Eq s.sym2 EmptyCollection.emptyCollection) (Eq s EmptyCollection.emptyC …
  -/
  rw [← val_eq_zero, sym2_val, Multiset.sym2_eq_zero_iff, val_eq_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem sym2_nonempty : s.sym2.Nonempty ↔ s.Nonempty := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff s.sym2.Nonempty s.Nonempty
  -/
  rw [← not_iff_not]
  /-
    α : Type u_1
    s : Finset α
    ⊢ Iff (Not s.sym2.Nonempty) (Not s.Nonempty)
  -/
  simp_rw [not_nonempty_iff_eq_empty, sym2_eq_empty]
  /-
    🎉 no goals
  -/


@[aesop safe apply (rule_sets := [finsetNonempty])]
protected alias ⟨_, Nonempty.sym2⟩ := sym2_nonempty


@[simp]
theorem sym2_singleton (a : α) : ({a} : Finset α).sym2 = {Sym2.diag a} := rfl


/-- Finset **stars and bars** for the case `n = 2`. -/
theorem card_sym2 (s : Finset α) : s.sym2.card = Nat.choose (s.card + 1) 2 := by
  /-
    α : Type u_1
    s : Finset α
    ⊢ Eq s.sym2.card ((HAdd.hAdd s.card 1).choose 2)
  -/
  rw [card_def, sym2_val, Multiset.card_sym2, ← card_def]
  /-
    🎉 no goals
  -/


theorem sym2_eq_image : s.sym2 = (s ×ˢ s).image Sym2.mk := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    ⊢ Eq s.sym2 (Finset.image Sym2.mk (SProd.sprod s s))
  -/
  ext z
  /-
    case h
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    z : Sym2 α
    ⊢ Iff (Membership.mem s.sym2 z) (Membership.mem (Finset.image Sym2.mk (SProd.s …
  -/
  refine z.ind fun x y ↦ ?_
  /-
    case h
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    z : Sym2 α
    x y : α
    ⊢ Iff (Membership.mem s.sym2 (Sym2.mk { fst := x, snd := y })) (Membership.mem …
  -/
  rw [mk_mem_sym2_iff, mem_image]
  /-
    case h
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    z : Sym2 α
    x y : α
    ⊢ Iff (And (Membership.mem s x) (Membership.mem s y)) (Exists fun a => And (Me …
  -/
  constructor
    /-
      case h.mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      z : Sym2 α
      x y : α
      ⊢ And (Membership.mem s x) (Membership.mem s y) → Exists fun a => And (Members …
    -/
  · intro h
    /-
      case h.mp
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      z : Sym2 α
      x y : α
      h : And (Membership.mem s x) (Membership.mem s y)
      ⊢ Exists fun a => And (Membership.mem (SProd.sprod s s) a) (Eq (Sym2.mk a) (Sy …
    -/
    use (x, y)
    /-
      case h
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      z : Sym2 α
      x y : α
      h : And (Membership.mem s x) (Membership.mem s y)
      ⊢ And (Membership.mem (SProd.sprod s s) { fst := x, snd := y }) (Eq (Sym2.mk { …
    -/
    simp only [mem_product, h, and_self, true_and]
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      z : Sym2 α
      x y : α
      ⊢ (Exists fun a => And (Membership.mem (SProd.sprod s s) a) (Eq (Sym2.mk a) (S …
    -/
  · rintro ⟨⟨a, b⟩, h⟩
    /-
      case h.mpr.intro.mk
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      z : Sym2 α
      x y a b : α
      h : And (Membership.mem (SProd.sprod s s) { fst := a, snd := b }) (Eq (Sym2.mk …
      ⊢ And (Membership.mem s x) (Membership.mem s y)
    -/
    simp only [mem_product, Sym2.eq_iff] at h
    /-
      case h.mpr.intro.mk
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      z : Sym2 α
      x y a b : α
      h : And (And (Membership.mem s a) (Membership.mem s b)) (Or (And (Eq a x) (Eq  …
      ⊢ And (Membership.mem s x) (Membership.mem s y)
    -/
    obtain ⟨h, (⟨rfl, rfl⟩ | ⟨rfl, rfl⟩)⟩ := h
          /-
            case h.mpr.intro.mk.intro.inl.intro
            α : Type u_1
            s : Finset α
            inst✝ : DecidableEq α
            z : Sym2 α
            a b : α
            h : And (Membership.mem s a) (Membership.mem s b)
            ⊢ And (Membership.mem s a) (Membership.mem s b)
          -/
          /-
            🎉 no goals
          -/
      <;> simp [h]
          /-
            🎉 no goals
          -/


theorem isDiag_mk_of_mem_diag {a : α × α} (h : a ∈ s.diag) : (Sym2.mk a).IsDiag :=
  (Sym2.isDiag_iff_proj_eq _).2 (mem_diag.1 h).2


theorem not_isDiag_mk_of_mem_offDiag {a : α × α} (h : a ∈ s.offDiag) :
    ¬ (Sym2.mk a).IsDiag := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    a : Prod α α
    h : Membership.mem s.offDiag a
    ⊢ Not (Sym2.mk a).IsDiag
  -/
  rw [Sym2.isDiag_iff_proj_eq]
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    a : Prod α α
    h : Membership.mem s.offDiag a
    ⊢ Not (Eq a.1 a.2)
  -/
  exact (mem_offDiag.1 h).2.2
  /-
    🎉 no goals
  -/


@[simp]
theorem diag_mem_sym2_mem_iff : (∀ b, b ∈ Sym2.diag a → b ∈ s) ↔ a ∈ s := by
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (∀ (b : α), Membership.mem (Sym2.diag a) b → Membership.mem s b) (Member …
  -/
  rw [← mem_sym2_iff]
  /-
    α : Type u_1
    s : Finset α
    a : α
    ⊢ Iff (Membership.mem s.sym2 (Sym2.diag a)) (Membership.mem s a)
  -/
  exact mk_mem_sym2_iff.trans <| and_self_iff
  /-
    🎉 no goals
  -/


                                                               /-
                                                                 α : Type u_1
                                                                 s : Finset α
                                                                 a : α
                                                                 ⊢ Iff (Membership.mem s.sym2 (Sym2.diag a)) (Membership.mem s a)
                                                               -/
theorem diag_mem_sym2_iff : Sym2.diag a ∈ s.sym2 ↔ a ∈ s := by simp [diag_mem_sym2_mem_iff]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem image_diag_union_image_offDiag [DecidableEq α] :
    s.diag.image Sym2.mk ∪ s.offDiag.image Sym2.mk = s.sym2 := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    ⊢ Eq (Union.union (Finset.image Sym2.mk s.diag) (Finset.image Sym2.mk s.offDia …
  -/
  rw [← image_union, diag_union_offDiag, sym2_eq_image]
  /-
    🎉 no goals
  -/


instance : DecidableEq (Sym α n) :=
  inferInstanceAs <| DecidableEq <| Subtype _


/-- Lifts a finset to `Sym α n`. `s.sym n` is the finset of all unordered tuples of cardinality `n`
with elements in `s`. -/
protected def sym (s : Finset α) : ∀ n, Finset (Sym α n)
  | 0 => {∅}
  | n + 1 => s.sup fun a ↦ Finset.image (Sym.cons a) (s.sym n)


@[simp]
theorem sym_zero : s.sym 0 = {∅} := rfl


@[simp]
theorem sym_succ : s.sym (n + 1) = s.sup fun a ↦ (s.sym n).image <| Sym.cons a := rfl


@[simp]
theorem mem_sym_iff {m : Sym α n} : m ∈ s.sym n ↔ ∀ a ∈ m, a ∈ s := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    n : Nat
    m : Sym α n
    ⊢ Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership.mem m a → Membership …
  -/
  induction' n with n ih
    /-
      case zero
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n : Nat
      m : Sym α 0
      ⊢ Iff (Membership.mem (s.sym 0) m) (∀ (a : α), Membership.mem m a → Membership …
    -/
  · refine mem_singleton.trans ⟨?_, fun _ ↦ Sym.eq_nil_of_card_zero _⟩
    /-
      case zero
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n : Nat
      m : Sym α 0
      ⊢ Eq m EmptyCollection.emptyCollection → ∀ (a : α), Membership.mem m a → Membe …
    -/
    rintro rfl
    /-
      case zero
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n : Nat
      ⊢ ∀ (a : α), Membership.mem EmptyCollection.emptyCollection a → Membership.mem …
    -/
    exact fun a ha ↦ (Finset.not_mem_empty _ ha).elim
    /-
      🎉 no goals
    -/
  /-
    case succ
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    n✝ n : Nat
    ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
    m : Sym α (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem (s.sym (HAdd.hAdd n 1)) m) (∀ (a : α), Membership.mem m  …
  -/
  refine mem_sup.trans ⟨?_, fun h ↦ ?_⟩
    /-
      case succ.refine_1
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ n : Nat
      ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
      m : Sym α (HAdd.hAdd n 1)
      ⊢ (Exists fun i => And (Membership.mem s i) (Membership.mem (Finset.image (Sym …
    -/
  · rintro ⟨a, ha, he⟩ b hb
    /-
      case succ.refine_1.intro.intro
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ n : Nat
      ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
      m : Sym α (HAdd.hAdd n 1)
      a : α
      ha : Membership.mem s a
      he : Membership.mem (Finset.image (Sym.cons a) (s.sym n)) m
      b : α
      hb : Membership.mem m b
      ⊢ Membership.mem s b
    -/
    rw [mem_image] at he
    /-
      case succ.refine_1.intro.intro
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ n : Nat
      ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
      m : Sym α (HAdd.hAdd n 1)
      a : α
      ha : Membership.mem s a
      he : Exists fun a_1 => And (Membership.mem (s.sym n) a_1) (Eq (Sym.cons a a_1) …
      b : α
      hb : Membership.mem m b
      ⊢ Membership.mem s b
    -/
    obtain ⟨m, he, rfl⟩ := he
    /-
      case succ.refine_1.intro.intro.intro.intro
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ n : Nat
      ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
      a : α
      ha : Membership.mem s a
      b : α
      m : Sym α n
      he : Membership.mem (s.sym n) m
      hb : Membership.mem (Sym.cons a m) b
      ⊢ Membership.mem s b
    -/
    rw [Sym.mem_cons] at hb
    /-
      case succ.refine_1.intro.intro.intro.intro
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ n : Nat
      ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
      a : α
      ha : Membership.mem s a
      b : α
      m : Sym α n
      he : Membership.mem (s.sym n) m
      hb : Or (Eq b a) (Membership.mem m b)
      ⊢ Membership.mem s b
    -/
    obtain rfl | hb := hb
      /-
        case succ.refine_1.intro.intro.intro.intro.inl
        α : Type u_1
        s : Finset α
        inst✝ : DecidableEq α
        n✝ n : Nat
        ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
        b : α
        m : Sym α n
        he : Membership.mem (s.sym n) m
        ha : Membership.mem s b
        ⊢ Membership.mem s b
      -/
    · exact ha
      /-
        🎉 no goals
      -/
      /-
        case succ.refine_1.intro.intro.intro.intro.inr
        α : Type u_1
        s : Finset α
        inst✝ : DecidableEq α
        n✝ n : Nat
        ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
        a : α
        ha : Membership.mem s a
        b : α
        m : Sym α n
        he : Membership.mem (s.sym n) m
        hb : Membership.mem m b
        ⊢ Membership.mem s b
      -/
    · exact ih.1 he _ hb
      /-
        🎉 no goals
      -/
    /-
      case succ.refine_2
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ n : Nat
      ih : ∀ {m : Sym α n}, Iff (Membership.mem (s.sym n) m) (∀ (a : α), Membership. …
      m : Sym α (HAdd.hAdd n 1)
      h : ∀ (a : α), Membership.mem m a → Membership.mem s a
      ⊢ Exists fun i => And (Membership.mem s i) (Membership.mem (Finset.image (Sym. …
    -/
  · obtain ⟨a, m, rfl⟩ := m.exists_eq_cons_of_succ
    exact
      ⟨a, h _ <| Sym.mem_cons_self _ _,
        mem_image_of_mem _ <| ih.2 fun b hb ↦ h _ <| Sym.mem_cons_of_mem hb⟩


@[simp]
theorem sym_empty (n : ℕ) : (∅ : Finset α).sym (n + 1) = ∅ := rfl


theorem replicate_mem_sym (ha : a ∈ s) (n : ℕ) : Sym.replicate n a ∈ s.sym n :=
                              /-
                                α : Type u_1
                                s : Finset α
                                a : α
                                inst✝ : DecidableEq α
                                ha : Membership.mem s a
                                n : Nat
                                b : α
                                hb : Membership.mem (Sym.replicate n a) b
                                ⊢ Membership.mem s b
                              -/
  mem_sym_iff.2 fun b hb ↦ by rwa [(Sym.mem_replicate.1 hb).2]
                              /-
                                🎉 no goals
                              -/


protected theorem Nonempty.sym (h : s.Nonempty) (n : ℕ) : (s.sym n).Nonempty :=
  let ⟨_a, ha⟩ := h
  ⟨_, replicate_mem_sym ha n⟩


@[simp]
theorem sym_singleton (a : α) (n : ℕ) : ({a} : Finset α).sym n = {Sym.replicate n a} :=
  eq_singleton_iff_unique_mem.2
    ⟨replicate_mem_sym (mem_singleton.2 rfl) _, fun _s hs ↦
      Sym.eq_replicate_iff.2 fun _b hb ↦ eq_of_mem_singleton <| mem_sym_iff.1 hs _ hb⟩


theorem eq_empty_of_sym_eq_empty (h : s.sym n = ∅) : s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    n : Nat
    h : Eq (s.sym n) EmptyCollection.emptyCollection
    ⊢ Eq s EmptyCollection.emptyCollection
  -/
  rw [← not_nonempty_iff_eq_empty] at h ⊢
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    n : Nat
    h : Not (s.sym n).Nonempty
    ⊢ Not s.Nonempty
  -/
  exact fun hs ↦ h (hs.sym _)
  /-
    🎉 no goals
  -/


@[simp]
theorem sym_eq_empty : s.sym n = ∅ ↔ n ≠ 0 ∧ s = ∅ := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    n : Nat
    ⊢ Iff (Eq (s.sym n) EmptyCollection.emptyCollection) (And (Ne n 0) (Eq s Empty …
  -/
  cases n
    /-
      case zero
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      ⊢ Iff (Eq (s.sym 0) EmptyCollection.emptyCollection) (And (Ne 0 0) (Eq s Empty …
    -/
  · exact iff_of_false (singleton_ne_empty _) fun h ↦ (h.1 rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case succ
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ : Nat
      ⊢ Iff (Eq (s.sym (HAdd.hAdd n✝ 1)) EmptyCollection.emptyCollection) (And (Ne ( …
    -/
  · refine ⟨fun h ↦ ⟨Nat.succ_ne_zero _, eq_empty_of_sym_eq_empty h⟩, ?_⟩
    /-
      case succ
      α : Type u_1
      s : Finset α
      inst✝ : DecidableEq α
      n✝ : Nat
      ⊢ And (Ne (HAdd.hAdd n✝ 1) 0) (Eq s EmptyCollection.emptyCollection) → Eq (s.s …
    -/
    rintro ⟨_, rfl⟩
    /-
      case succ.intro
      α : Type u_1
      inst✝ : DecidableEq α
      n✝ : Nat
      left✝ : Ne (HAdd.hAdd n✝ 1) 0
      ⊢ Eq (EmptyCollection.emptyCollection.sym (HAdd.hAdd n✝ 1)) EmptyCollection.em …
    -/
    exact sym_empty _
    /-
      🎉 no goals
    -/


@[simp]
theorem sym_nonempty : (s.sym n).Nonempty ↔ n = 0 ∨ s.Nonempty := by
  /-
    α : Type u_1
    s : Finset α
    inst✝ : DecidableEq α
    n : Nat
    ⊢ Iff (s.sym n).Nonempty (Or (Eq n 0) s.Nonempty)
  -/
  simp only [nonempty_iff_ne_empty, ne_eq, sym_eq_empty, not_and_or, not_ne_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem sym_univ [Fintype α] (n : ℕ) : (univ : Finset α).sym n = univ :=
  eq_univ_iff_forall.2 fun _s ↦ mem_sym_iff.2 fun _a _ ↦ mem_univ _


@[simp]
theorem sym_mono (h : s ⊆ t) (n : ℕ) : s.sym n ⊆ t.sym n := fun _m hm ↦
  mem_sym_iff.2 fun _a ha ↦ h <| mem_sym_iff.1 hm _ ha


@[simp]
theorem sym_inter (s t : Finset α) (n : ℕ) : (s ∩ t).sym n = s.sym n ∩ t.sym n := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    n : Nat
    ⊢ Eq ((Inter.inter s t).sym n) (Inter.inter (s.sym n) (t.sym n))
  -/
  ext m
  /-
    case h
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    n : Nat
    m : Sym α n
    ⊢ Iff (Membership.mem ((Inter.inter s t).sym n) m) (Membership.mem (Inter.inte …
  -/
  simp only [mem_inter, mem_sym_iff, imp_and, forall_and]
  /-
    🎉 no goals
  -/


@[simp]
theorem sym_union (s t : Finset α) (n : ℕ) : s.sym n ∪ t.sym n ⊆ (s ∪ t).sym n :=
  union_subset (sym_mono subset_union_left n) (sym_mono subset_union_right n)


theorem sym_fill_mem (a : α) {i : Fin (n + 1)} {m : Sym α (n - i)} (h : m ∈ s.sym (n - i)) :
    m.fill a i ∈ (insert a s).sym n :=
  mem_sym_iff.2 fun b hb ↦
    mem_insert.2 <| (Sym.mem_fill_iff.1 hb).imp And.right <| mem_sym_iff.1 h b


theorem sym_filterNe_mem {m : Sym α n} (a : α) (h : m ∈ s.sym n) :
    (m.filterNe a).2 ∈ (Finset.erase s a).sym (n - (m.filterNe a).1) :=
  mem_sym_iff.2 fun b H ↦
    mem_erase.2 <| (Multiset.mem_filter.1 H).symm.imp Ne.symm <| mem_sym_iff.1 h b


/-- If `a` does not belong to the finset `s`, then the `n`th symmetric power of `{a} ∪ s` is
  in 1-1 correspondence with the disjoint union of the `n - i`th symmetric powers of `s`,
  for `0 ≤ i ≤ n`. -/
@[simps]
def symInsertEquiv (h : a ∉ s) : (insert a s).sym n ≃ Σi : Fin (n + 1), s.sym (n - i) where
                                        /-
                                          α : Type u_1
                                          β : Type u_2
                                          s t : Finset α
                                          a b : α
                                          inst✝ : DecidableEq α
                                          n : Nat
                                          h : Not (Membership.mem s a)
                                          m : Subtype fun x => Membership.mem ((Insert.insert a s).sym n) x
                                          ⊢ Membership.mem (s.sym (HSub.hSub n ↑(Sym.filterNe a ↑m).fst)) (Sym.filterNe  …
                                        -/
  toFun m := ⟨_, (m.1.filterNe a).2, by convert sym_filterNe_mem a m.2; rw [erase_insert h]⟩
                                                                        /-
                                                                          🎉 no goals
                                                                        -/
  invFun m := ⟨m.2.1.fill a m.1, sym_fill_mem a m.2.2⟩
  left_inv m := Subtype.ext <| m.1.fill_filterNe a
  right_inv := fun ⟨i, m, hm⟩ ↦ by
    refine Function.Injective.sigma_map (β₂ := ?_) (f₂ := ?_)
        (Function.injective_id) (fun i ↦ ?_) ?_
      /-
        case refine_1
        α : Type u_1
        β : Type u_2
        s t : Finset α
        a b : α
        inst✝ : DecidableEq α
        n : Nat
        h : Not (Membership.mem s a)
        x✝ : Sigma fun i => Subtype fun x => Membership.mem (s.sym (HSub.hSub n ↑i)) x
        i : Fin (HAdd.hAdd n 1)
        m : Sym α (HSub.hSub n ↑i)
        hm : Membership.mem (s.sym (HSub.hSub n ↑i)) m
        ⊢ Fin (HAdd.hAdd n 1) → Type ?u.67306
      -/
    · exact fun i ↦ Sym α (n - i)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      s t : Finset α
      a b : α
      inst✝ : DecidableEq α
      n : Nat
      h : Not (Membership.mem s a)
      x✝ : Sigma fun i => Subtype fun x => Membership.mem (s.sym (HSub.hSub n ↑i)) x
      i : Fin (HAdd.hAdd n 1)
      m : Sym α (HSub.hSub n ↑i)
      hm : Membership.mem (s.sym (HSub.hSub n ↑i)) m
      ⊢ (a : Fin (HAdd.hAdd n 1)) → (Subtype fun x => Membership.mem (s.sym (HSub.hS …
    -/
    swap
      /-
        case refine_3
        α : Type u_1
        β : Type u_2
        s t : Finset α
        a b : α
        inst✝ : DecidableEq α
        n : Nat
        h : Not (Membership.mem s a)
        x✝ : Sigma fun i => Subtype fun x => Membership.mem (s.sym (HSub.hSub n ↑i)) x
        i✝ : Fin (HAdd.hAdd n 1)
        m : Sym α (HSub.hSub n ↑i✝)
        hm : Membership.mem (s.sym (HSub.hSub n ↑i✝)) m
        i : Fin (HAdd.hAdd n 1)
        ⊢ Function.Injective (?refine_2 i)
      -/
    · exact Subtype.coe_injective
      /-
        🎉 no goals
      -/
    /-
      case refine_4
      α : Type u_1
      β : Type u_2
      s t : Finset α
      a b : α
      inst✝ : DecidableEq α
      n : Nat
      h : Not (Membership.mem s a)
      x✝ : Sigma fun i => Subtype fun x => Membership.mem (s.sym (HSub.hSub n ↑i)) x
      i : Fin (HAdd.hAdd n 1)
      m : Sym α (HSub.hSub n ↑i)
      hm : Membership.mem (s.sym (HSub.hSub n ↑i)) m
      ⊢ Eq (Sigma.map id (fun i a => ↑a) ((fun m => ⟨(Sym.filterNe a ↑m).fst, ⟨(Sym. …
    -/
    refine Eq.trans ?_ (Sym.filter_ne_fill a _ ?_)
    /-
      case refine_4.refine_1
      α : Type u_1
      β : Type u_2
      s t : Finset α
      a b : α
      inst✝ : DecidableEq α
      n : Nat
      h : Not (Membership.mem s a)
      x✝ : Sigma fun i => Subtype fun x => Membership.mem (s.sym (HSub.hSub n ↑i)) x
      i : Fin (HAdd.hAdd n 1)
      m : Sym α (HSub.hSub n ↑i)
      hm : Membership.mem (s.sym (HSub.hSub n ↑i)) m
      ⊢ Eq (Sigma.map id (fun i a => ↑a) ((fun m => ⟨(Sym.filterNe a ↑m).fst, ⟨(Sym. …
    -/
    exacts [rfl, h ∘ mem_sym_iff.1 hm a]
    /-
      🎉 no goals
    -/


