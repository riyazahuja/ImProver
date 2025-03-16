/-- Enumerator function for an unbounded set of ordinals. -/
noncomputable def enumOrd (s : Set Ordinal.{u}) (o : Ordinal.{u}) : Ordinal.{u} :=
  sInf (s ∩ { b | ∀ c, c < o → enumOrd s c < b })
termination_by o


@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem enumOrd_def (o : Ordinal.{u}) :
    enumOrd s o = sInf (s ∩ { b | ∀ c, c < o → enumOrd s c < b }) := by
  /-
    s : Set Ordinal.{u}
    o : Ordinal.{u}
    ⊢ Eq (Ordinal.enumOrd s o) (InfSet.sInf (Inter.inter s (setOf fun b => ∀ (c :  …
  -/
  rw [enumOrd]
  /-
    🎉 no goals
  -/


theorem enumOrd_le_of_forall_lt (ha : a ∈ s) (H : ∀ b < o, enumOrd s b < a) : enumOrd s o ≤ a := by
  /-
    o a : Ordinal.{u}
    s : Set Ordinal.{u}
    ha : Membership.mem s a
    H : ∀ (b : Ordinal.{u}), LT.lt b o → LT.lt (Ordinal.enumOrd s b) a
    ⊢ LE.le (Ordinal.enumOrd s o) a
  -/
  rw [enumOrd]
  /-
    o a : Ordinal.{u}
    s : Set Ordinal.{u}
    ha : Membership.mem s a
    H : ∀ (b : Ordinal.{u}), LT.lt b o → LT.lt (Ordinal.enumOrd s b) a
    ⊢ LE.le (InfSet.sInf (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt …
  -/
  exact csInf_le' ⟨ha, H⟩
  /-
    🎉 no goals
  -/


/-- The set in the definition of `enumOrd` is nonempty. -/
private theorem enumOrd_nonempty (hs : ¬ BddAbove s) (o : Ordinal) :
    (s ∩ { b | ∀ c, c < o → enumOrd s c < b }).Nonempty := by
  /-
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    o : Ordinal.{u}
    ⊢ (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c o → LT.lt (Ordin …
  -/
  rw [not_bddAbove_iff] at hs
  /-
    s : Set Ordinal.{u}
    hs : ∀ (x : Ordinal.{u}), Exists fun y => And (Membership.mem s y) (LT.lt x y)
    o : Ordinal.{u}
    ⊢ (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c o → LT.lt (Ordin …
  -/
  obtain ⟨a, ha⟩ := bddAbove_of_small (enumOrd s '' Iio o)
  /-
    case intro
    s : Set Ordinal.{u}
    hs : ∀ (x : Ordinal.{u}), Exists fun y => And (Membership.mem s y) (LT.lt x y)
    o a : Ordinal.{u}
    ha : Membership.mem (upperBounds (Set.image (Ordinal.enumOrd s) (Set.Iio o))) a
    ⊢ (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c o → LT.lt (Ordin …
  -/
  obtain ⟨b, hb, hba⟩ := hs a
  /-
    case intro.intro.intro
    s : Set Ordinal.{u}
    hs : ∀ (x : Ordinal.{u}), Exists fun y => And (Membership.mem s y) (LT.lt x y)
    o a : Ordinal.{u}
    ha : Membership.mem (upperBounds (Set.image (Ordinal.enumOrd s) (Set.Iio o))) a
    b : Ordinal.{u}
    hb : Membership.mem s b
    hba : LT.lt a b
    ⊢ (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c o → LT.lt (Ordin …
  -/
  exact ⟨b, hb, fun c hc ↦ (ha (mem_image_of_mem _ hc)).trans_lt hba⟩
  /-
    🎉 no goals
  -/


private theorem enumOrd_mem_aux (hs : ¬ BddAbove s) (o : Ordinal) :
    enumOrd s o ∈ s ∩ { b | ∀ c, c < o → enumOrd s c < b } := by
  /-
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    o : Ordinal.{u}
    ⊢ Membership.mem (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c o …
  -/
  rw [enumOrd]
  /-
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    o : Ordinal.{u}
    ⊢ Membership.mem (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c o …
  -/
  exact csInf_mem (enumOrd_nonempty hs o)
  /-
    🎉 no goals
  -/


theorem enumOrd_mem (hs : ¬ BddAbove s) (o : Ordinal) : enumOrd s o ∈ s :=
  (enumOrd_mem_aux hs o).1


theorem enumOrd_strictMono (hs : ¬ BddAbove s) : StrictMono (enumOrd s) :=
  fun a b ↦ (enumOrd_mem_aux hs b).2 a


theorem enumOrd_injective (hs : ¬ BddAbove s) : Function.Injective (enumOrd s) :=
  (enumOrd_strictMono hs).injective


theorem enumOrd_inj (hs : ¬ BddAbove s) {a b : Ordinal} : enumOrd s a = enumOrd s b ↔ a = b :=
  (enumOrd_injective hs).eq_iff


theorem enumOrd_le_enumOrd (hs : ¬ BddAbove s) {a b : Ordinal} :
    enumOrd s a ≤ enumOrd s b ↔ a ≤ b :=
  (enumOrd_strictMono hs).le_iff_le


theorem enumOrd_lt_enumOrd (hs : ¬ BddAbove s) {a b : Ordinal} :
    enumOrd s a < enumOrd s b ↔ a < b :=
  (enumOrd_strictMono hs).lt_iff_lt


theorem id_le_enumOrd (hs : ¬ BddAbove s) : id ≤ enumOrd s :=
  (enumOrd_strictMono hs).id_le


theorem le_enumOrd_self (hs : ¬ BddAbove s) {a} : a ≤ enumOrd s a :=
  (enumOrd_strictMono hs).le_apply


theorem enumOrd_succ_le (hs : ¬ BddAbove s) (ha : a ∈ s) (hb : enumOrd s b < a) :
    enumOrd s (succ b) ≤ a := by
  /-
    a b : Ordinal.{u}
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    ha : Membership.mem s a
    hb : LT.lt (Ordinal.enumOrd s b) a
    ⊢ LE.le (Ordinal.enumOrd s (Order.succ b)) a
  -/
  apply enumOrd_le_of_forall_lt ha
  /-
    a b : Ordinal.{u}
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    ha : Membership.mem s a
    hb : LT.lt (Ordinal.enumOrd s b) a
    ⊢ ∀ (b_1 : Ordinal.{u}), LT.lt b_1 (Order.succ b) → LT.lt (Ordinal.enumOrd s b …
  -/
  intro c hc
  /-
    a b : Ordinal.{u}
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    ha : Membership.mem s a
    hb : LT.lt (Ordinal.enumOrd s b) a
    c : Ordinal.{u}
    hc : LT.lt c (Order.succ b)
    ⊢ LT.lt (Ordinal.enumOrd s c) a
  -/
  rw [lt_succ_iff] at hc
  /-
    a b : Ordinal.{u}
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    ha : Membership.mem s a
    hb : LT.lt (Ordinal.enumOrd s b) a
    c : Ordinal.{u}
    hc : LE.le c b
    ⊢ LT.lt (Ordinal.enumOrd s c) a
  -/
  exact ((enumOrd_strictMono hs).monotone hc).trans_lt hb
  /-
    🎉 no goals
  -/


theorem range_enumOrd (hs : ¬ BddAbove s) : range (enumOrd s) = s := by
  /-
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    ⊢ Eq (Set.range (Ordinal.enumOrd s)) s
  -/
  ext a
  /-
    case h
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    a : Ordinal.{u}
    ⊢ Iff (Membership.mem (Set.range (Ordinal.enumOrd s)) a) (Membership.mem s a)
  -/
  let t := { b | a ≤ enumOrd s b }
  /-
    case h
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    a : Ordinal.{u}
    t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
    ⊢ Iff (Membership.mem (Set.range (Ordinal.enumOrd s)) a) (Membership.mem s a)
  -/
  constructor
    /-
      case h.mp
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      a : Ordinal.{u}
      t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
      ⊢ Membership.mem (Set.range (Ordinal.enumOrd s)) a → Membership.mem s a
    -/
  · rintro ⟨b, rfl⟩
    /-
      case h.mp.intro
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      b : Ordinal.{u}
      t : Set Ordinal.{u} := setOf fun b_1 => LE.le (Ordinal.enumOrd s b) (Ordinal.e …
      ⊢ Membership.mem s (Ordinal.enumOrd s b)
    -/
    exact enumOrd_mem hs b
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      a : Ordinal.{u}
      t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
      ⊢ Membership.mem s a → Membership.mem (Set.range (Ordinal.enumOrd s)) a
    -/
  · intro ha
    /-
      case h.mpr
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      a : Ordinal.{u}
      t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
      ha : Membership.mem s a
      ⊢ Membership.mem (Set.range (Ordinal.enumOrd s)) a
    -/
    refine ⟨sInf t, (enumOrd_le_of_forall_lt ha ?_).antisymm ?_⟩
      /-
        case h.mpr.refine_1
        s : Set Ordinal.{u}
        hs : Not (BddAbove s)
        a : Ordinal.{u}
        t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
        ha : Membership.mem s a
        ⊢ ∀ (b : Ordinal.{u}), LT.lt b (InfSet.sInf t) → LT.lt (Ordinal.enumOrd s b) a
      -/
    · intro b hb
      /-
        case h.mpr.refine_1
        s : Set Ordinal.{u}
        hs : Not (BddAbove s)
        a : Ordinal.{u}
        t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
        ha : Membership.mem s a
        b : Ordinal.{u}
        hb : LT.lt b (InfSet.sInf t)
        ⊢ LT.lt (Ordinal.enumOrd s b) a
      -/
      by_contra! hb'
      /-
        case h.mpr.refine_1
        s : Set Ordinal.{u}
        hs : Not (BddAbove s)
        a : Ordinal.{u}
        t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
        ha : Membership.mem s a
        b : Ordinal.{u}
        hb : LT.lt b (InfSet.sInf t)
        hb' : LE.le a (Ordinal.enumOrd s b)
        ⊢ False
      -/
      exact hb.not_le (csInf_le' hb')
      /-
        🎉 no goals
      -/
      /-
        case h.mpr.refine_2
        s : Set Ordinal.{u}
        hs : Not (BddAbove s)
        a : Ordinal.{u}
        t : Set Ordinal.{u} := setOf fun b => LE.le a (Ordinal.enumOrd s b)
        ha : Membership.mem s a
        ⊢ LE.le a (Ordinal.enumOrd s (InfSet.sInf t))
      -/
    · exact csInf_mem (s := t) ⟨a, (enumOrd_strictMono hs).id_le a⟩
      /-
        🎉 no goals
      -/


theorem enumOrd_surjective (hs : ¬ BddAbove s) {b : Ordinal} (hb : b ∈ s) :
    ∃ a, enumOrd s a = b := by
  /-
    s : Set Ordinal.{u}
    hs : Not (BddAbove s)
    b : Ordinal.{u}
    hb : Membership.mem s b
    ⊢ Exists fun a => Eq (Ordinal.enumOrd s a) b
  -/
  rwa [← range_enumOrd hs] at hb
  /-
    🎉 no goals
  -/


theorem enumOrd_le_of_subset {t : Set Ordinal} (hs : ¬ BddAbove s) (hst : s ⊆ t) :
    enumOrd t ≤ enumOrd s := by
  /-
    s t : Set Ordinal.{u}
    hs : Not (BddAbove s)
    hst : HasSubset.Subset s t
    ⊢ LE.le (Ordinal.enumOrd t) (Ordinal.enumOrd s)
  -/
  intro a
  /-
    s t : Set Ordinal.{u}
    hs : Not (BddAbove s)
    hst : HasSubset.Subset s t
    a : Ordinal.{u}
    ⊢ LE.le (Ordinal.enumOrd t a) (Ordinal.enumOrd s a)
  -/
  rw [enumOrd, enumOrd]
  /-
    s t : Set Ordinal.{u}
    hs : Not (BddAbove s)
    hst : HasSubset.Subset s t
    a : Ordinal.{u}
    ⊢ LE.le (InfSet.sInf (Inter.inter t (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt …
  -/
  apply csInf_le_csInf' (enumOrd_nonempty hs a) (inter_subset_inter hst _)
  /-
    s t : Set Ordinal.{u}
    hs : Not (BddAbove s)
    hst : HasSubset.Subset s t
    a : Ordinal.{u}
    ⊢ HasSubset.Subset (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c a → LT.lt (Ord …
  -/
  intro b hb c hc
  /-
    s t : Set Ordinal.{u}
    hs : Not (BddAbove s)
    hst : HasSubset.Subset s t
    a b : Ordinal.{u}
    hb : Membership.mem (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c a → LT.lt (Or …
    c : Ordinal.{u}
    hc : LT.lt c a
    ⊢ LT.lt (Ordinal.enumOrd t c) b
  -/
  exact (enumOrd_le_of_subset hs hst c).trans_lt <| hb c hc
  /-
    🎉 no goals
  -/
termination_by a => a


/-- A characterization of `enumOrd`: it is the unique strict monotonic function with range `s`. -/
theorem eq_enumOrd (f : Ordinal → Ordinal) (hs : ¬ BddAbove s) :
    enumOrd s = f ↔ StrictMono f ∧ range f = s := by
  /-
    s : Set Ordinal.{u}
    f : Ordinal.{u} → Ordinal.{u}
    hs : Not (BddAbove s)
    ⊢ Iff (Eq (Ordinal.enumOrd s) f) (And (StrictMono f) (Eq (Set.range f) s))
  -/
  constructor
    /-
      case mp
      s : Set Ordinal.{u}
      f : Ordinal.{u} → Ordinal.{u}
      hs : Not (BddAbove s)
      ⊢ Eq (Ordinal.enumOrd s) f → And (StrictMono f) (Eq (Set.range f) s)
    -/
  · rintro rfl
    /-
      case mp
      s : Set Ordinal.{u}
      hs : Not (BddAbove s)
      ⊢ And (StrictMono (Ordinal.enumOrd s)) (Eq (Set.range (Ordinal.enumOrd s)) s)
    -/
    exact ⟨enumOrd_strictMono hs, range_enumOrd hs⟩
    /-
      🎉 no goals
    -/
    /-
      case mpr
      s : Set Ordinal.{u}
      f : Ordinal.{u} → Ordinal.{u}
      hs : Not (BddAbove s)
      ⊢ And (StrictMono f) (Eq (Set.range f) s) → Eq (Ordinal.enumOrd s) f
    -/
  · rintro ⟨h₁, h₂⟩
    /-
      case mpr.intro
      s : Set Ordinal.{u}
      f : Ordinal.{u} → Ordinal.{u}
      hs : Not (BddAbove s)
      h₁ : StrictMono f
      h₂ : Eq (Set.range f) s
      ⊢ Eq (Ordinal.enumOrd s) f
    -/
    rwa [← (enumOrd_strictMono hs).range_inj h₁, range_enumOrd hs, eq_comm]
    /-
      🎉 no goals
    -/


theorem enumOrd_range {f : Ordinal → Ordinal} (hf : StrictMono f) : enumOrd (range f) = f :=
  (eq_enumOrd _ hf.not_bddAbove_range_of_wellFoundedLT).2 ⟨hf, rfl⟩


@[simp]
theorem enumOrd_univ : enumOrd Set.univ = id := by
  /-
    ⊢ Eq (Ordinal.enumOrd Set.univ) id
  -/
  rw [← range_id]
  /-
    ⊢ Eq (Ordinal.enumOrd (Set.range id)) id
  -/
  exact enumOrd_range strictMono_id
  /-
    🎉 no goals
  -/


@[simp]
theorem enumOrd_zero : enumOrd s 0 = sInf s := by
  /-
    s : Set Ordinal.{u}
    ⊢ Eq (Ordinal.enumOrd s 0) (InfSet.sInf s)
  -/
  rw [enumOrd]
  /-
    s : Set Ordinal.{u}
    ⊢ Eq (InfSet.sInf (Inter.inter s (setOf fun b => ∀ (c : Ordinal.{u}), LT.lt c  …
  -/
  simp [Ordinal.not_lt_zero]
  /-
    🎉 no goals
  -/


/-- An order isomorphism between an unbounded set of ordinals and the ordinals. -/
noncomputable def enumOrdOrderIso (s : Set Ordinal) (hs : ¬ BddAbove s) : Ordinal ≃o s :=
  StrictMono.orderIsoOfSurjective (fun o => ⟨_, enumOrd_mem hs o⟩) (enumOrd_strictMono hs) fun s =>
    let ⟨a, ha⟩ := enumOrd_surjective hs s.prop
    ⟨a, Subtype.eq ha⟩


