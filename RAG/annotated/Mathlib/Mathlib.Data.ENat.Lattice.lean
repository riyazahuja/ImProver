noncomputable instance : CompleteLinearOrder ENat :=
  inferInstanceAs (CompleteLinearOrder (WithTop ℕ))


noncomputable instance : CompleteLinearOrder (WithBot ENat) :=
  inferInstanceAs (CompleteLinearOrder (WithBot (WithTop ℕ)))


lemma iSup_coe_eq_top : ⨆ i, (f i : ℕ∞) = ⊤ ↔ ¬ BddAbove (range f) := WithTop.iSup_coe_eq_top

lemma iSup_coe_ne_top : ⨆ i, (f i : ℕ∞) ≠ ⊤ ↔ BddAbove (range f) := iSup_coe_eq_top.not_left

lemma iSup_coe_lt_top : ⨆ i, (f i : ℕ∞) < ⊤ ↔ BddAbove (range f) := WithTop.iSup_coe_lt_top

lemma iInf_coe_eq_top : ⨅ i, (f i : ℕ∞) = ⊤ ↔ IsEmpty ι := WithTop.iInf_coe_eq_top

lemma iInf_coe_ne_top : ⨅ i, (f i : ℕ∞) ≠ ⊤ ↔ Nonempty ι := by
  /-
    ι : Sort u_1
    f : ι → Nat
    ⊢ Iff (Ne (iInf fun i => ↑(f i)) Top.top) (Nonempty ι)
  -/
  rw [Ne, iInf_coe_eq_top, not_isEmpty_iff]
  /-
    🎉 no goals
  -/

lemma iInf_coe_lt_top : ⨅ i, (f i : ℕ∞) < ⊤ ↔ Nonempty ι := WithTop.iInf_coe_lt_top


lemma coe_sSup : BddAbove s → ↑(sSup s) = ⨆ a ∈ s, (a : ℕ∞) := WithTop.coe_sSup


lemma coe_sInf (hs : s.Nonempty) : ↑(sInf s) = ⨅ a ∈ s, (a : ℕ∞) :=
  WithTop.coe_sInf hs (OrderBot.bddBelow s)


lemma coe_iSup : BddAbove (range f) → ↑(⨆ i, f i) = ⨆ i, (f i : ℕ∞) := WithTop.coe_iSup _


@[norm_cast] lemma coe_iInf [Nonempty ι] : ↑(⨅ i, f i) = ⨅ i, (f i : ℕ∞) :=
  WithTop.coe_iInf (OrderBot.bddBelow _)


@[simp]
lemma iInf_eq_top_of_isEmpty [IsEmpty ι] : ⨅ i, (f i : ℕ∞) = ⊤ :=
  iInf_coe_eq_top.mpr ‹_›


lemma iInf_toNat : (⨅ i, (f i : ℕ∞)).toNat = ⨅ i, f i := by
  /-
    ι : Sort u_1
    f : ι → Nat
    ⊢ Eq (iInf fun i => ↑(f i)).toNat (iInf fun i => f i)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      f : ι → Nat
      h✝ : IsEmpty ι
      ⊢ Eq (iInf fun i => ↑(f i)).toNat (iInf fun i => f i)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      f : ι → Nat
      h✝ : Nonempty ι
      ⊢ Eq (iInf fun i => ↑(f i)).toNat (iInf fun i => f i)
    -/
  · norm_cast
    /-
      🎉 no goals
    -/


lemma iInf_eq_zero : ⨅ i, (f i : ℕ∞) = 0 ↔ ∃ i, f i = 0 := by
  /-
    ι : Sort u_1
    f : ι → Nat
    ⊢ Iff (Eq (iInf fun i => ↑(f i)) 0) (Exists fun i => Eq (f i) 0)
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_1
      f : ι → Nat
      h✝ : IsEmpty ι
      ⊢ Iff (Eq (iInf fun i => ↑(f i)) 0) (Exists fun i => Eq (f i) 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_1
      f : ι → Nat
      h✝ : Nonempty ι
      ⊢ Iff (Eq (iInf fun i => ↑(f i)) 0) (Exists fun i => Eq (f i) 0)
    -/
  · norm_cast
    /-
      case inr
      ι : Sort u_1
      f : ι → Nat
      h✝ : Nonempty ι
      ⊢ Iff (Eq (iInf fun i => f i) 0) (Exists fun i => Eq (f i) 0)
    -/
    rw [iInf, Nat.sInf_eq_zero]
    /-
      case inr
      ι : Sort u_1
      f : ι → Nat
      h✝ : Nonempty ι
      ⊢ Iff (Or (Membership.mem (Set.range fun i => f i) 0) (Eq (Set.range fun i =>  …
    -/
    exact ⟨fun h ↦ by simp_all, .inl⟩
    /-
      🎉 no goals
    -/


lemma sSup_eq_zero : sSup s = 0 ↔ ∀ a ∈ s, a = 0 :=
  sSup_eq_bot


lemma sInf_eq_zero : sInf s = 0 ↔ 0 ∈ s := by
  /-
    s : Set ENat
    ⊢ Iff (Eq (InfSet.sInf s) 0) (Membership.mem s 0)
  -/
  rw [← lt_one_iff_eq_zero]
  /-
    s : Set ENat
    ⊢ Iff (LT.lt (InfSet.sInf s) 1) (Membership.mem s 0)
  -/
  simp only [sInf_lt_iff, lt_one_iff_eq_zero, exists_eq_right]
  /-
    🎉 no goals
  -/


lemma sSup_eq_zero' : sSup s = 0 ↔ s = ∅ ∨ s = {0} :=
  sSup_eq_bot'


@[simp] lemma iSup_eq_zero : iSup f = 0 ↔ ∀ i, f i = 0 := iSup_eq_bot

                                                      /-
                                                        ι : Sort u_1
                                                        ⊢ Eq (iSup fun x => 0) 0
                                                      -/
@[simp] lemma iSup_zero : ⨆ _ : ι, (0 : ℕ∞) = 0 := by simp
                                                      /-
                                                        🎉 no goals
                                                      -/


lemma sSup_eq_top_of_infinite (h : s.Infinite) : sSup s = ⊤ := by
  /-
    s : Set ENat
    h : s.Infinite
    ⊢ Eq (SupSet.sSup s) Top.top
  -/
  apply (sSup_eq_top ..).mpr
  /-
    s : Set ENat
    h : s.Infinite
    ⊢ ∀ (b : ENat), LT.lt b Top.top → Exists fun a => And (Membership.mem s a) (LT …
  -/
  intro x hx
  cases x with
  | top => simp at hx
  | coe x =>
    contrapose! h
    simp only [not_infinite]
    apply Finite.subset <| Finite.Set.finite_image {n : ℕ | n ≤ x} (fun (n : ℕ) => (n : ℕ∞))
    intro y hy
    specialize h y hy
    have hxt : y < ⊤ := lt_of_le_of_lt h hx
    use y.toNat
    simp [toNat_le_of_le_coe h, LT.lt.ne_top hxt]


lemma finite_of_sSup_lt_top (h : sSup s < ⊤) : s.Finite := by
  /-
    s : Set ENat
    h : LT.lt (SupSet.sSup s) Top.top
    ⊢ s.Finite
  -/
  contrapose! h
  /-
    s : Set ENat
    h : Not s.Finite
    ⊢ LE.le Top.top (SupSet.sSup s)
  -/
  simp only [top_le_iff]
  /-
    s : Set ENat
    h : Not s.Finite
    ⊢ Eq (SupSet.sSup s) Top.top
  -/
  exact sSup_eq_top_of_infinite h
  /-
    🎉 no goals
  -/


lemma sSup_mem_of_nonempty_of_lt_top [Nonempty s] (hs' : sSup s < ⊤) : sSup s ∈ s :=
  Nonempty.csSup_mem .of_subtype (finite_of_sSup_lt_top hs')


lemma exists_eq_iSup_of_lt_top [Nonempty ι] (h : ⨆ i, f i < ⊤) :
    ∃ i, f i = ⨆ i, f i :=
  sSup_mem_of_nonempty_of_lt_top h


lemma exists_eq_iSup₂_of_lt_top {ι₁ ι₂ : Type*} {f : ι₁ → ι₂ → ℕ∞} [Nonempty ι₁] [Nonempty ι₂]
    (h : ⨆ i, ⨆ j, f i j < ⊤) : ∃ i j, f i j = ⨆ i, ⨆ j, f i j := by
  /-
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → ι₂ → ENat
    inst✝¹ : Nonempty ι₁
    inst✝ : Nonempty ι₂
    h : LT.lt (iSup fun i => iSup fun j => f i j) Top.top
    ⊢ Exists fun i => Exists fun j => Eq (f i j) (iSup fun i => iSup fun j => f i j)
  -/
  rw [iSup_prod'] at h ⊢
  /-
    ι₁ : Type u_2
    ι₂ : Type u_3
    f : ι₁ → ι₂ → ENat
    inst✝¹ : Nonempty ι₁
    inst✝ : Nonempty ι₂
    h : LT.lt (iSup fun x => f x.1 x.2) Top.top
    ⊢ Exists fun i => Exists fun j => Eq (f i j) (iSup fun x => f x.1 x.2)
  -/
  exact Prod.exists'.mp (exists_eq_iSup_of_lt_top h)
  /-
    🎉 no goals
  -/


lemma iSup_natCast : ⨆ n : ℕ, (n : ℕ∞) = ⊤ :=
  (iSup_eq_top _).2 fun _b hb ↦ ENat.exists_nat_gt (lt_top_iff_ne_top.1 hb)


lemma add_iSup [Nonempty ι] (f : ι → ℕ∞) : a + ⨆ i, f i = ⨆ i, a + f i := by
  /-
    ι : Sort u_2
    a : ENat
    inst✝ : Nonempty ι
    f : ι → ENat
    ⊢ Eq (HAdd.hAdd a (iSup fun i => f i)) (iSup fun i => HAdd.hAdd a (f i))
  -/
  obtain rfl | ha := eq_or_ne a ⊤
    /-
      case inl
      ι : Sort u_2
      inst✝ : Nonempty ι
      f : ι → ENat
      ⊢ Eq (HAdd.hAdd Top.top (iSup fun i => f i)) (iSup fun i => HAdd.hAdd Top.top  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Sort u_2
    a : ENat
    inst✝ : Nonempty ι
    f : ι → ENat
    ha : Ne a Top.top
    ⊢ Eq (HAdd.hAdd a (iSup fun i => f i)) (iSup fun i => HAdd.hAdd a (f i))
  -/
  refine le_antisymm ?_ <| iSup_le fun i ↦ add_le_add_left (le_iSup ..) _
  /-
    case inr
    ι : Sort u_2
    a : ENat
    inst✝ : Nonempty ι
    f : ι → ENat
    ha : Ne a Top.top
    ⊢ LE.le (HAdd.hAdd a (iSup fun i => f i)) (iSup fun i => HAdd.hAdd a (f i))
  -/
  refine add_le_of_le_tsub_left_of_le (le_iSup_of_le (Classical.arbitrary _) le_self_add) ?_
  /-
    case inr
    ι : Sort u_2
    a : ENat
    inst✝ : Nonempty ι
    f : ι → ENat
    ha : Ne a Top.top
    ⊢ LE.le (iSup fun i => f i) (HSub.hSub (iSup fun i => HAdd.hAdd a (f i)) a)
  -/
  exact iSup_le fun i ↦ ENat.le_sub_of_add_le_left ha <| le_iSup (a + f ·) i
  /-
    🎉 no goals
  -/


lemma iSup_add [Nonempty ι] (f : ι → ℕ∞) : (⨆ i, f i) + a = ⨆ i, f i + a := by
  /-
    ι : Sort u_2
    a : ENat
    inst✝ : Nonempty ι
    f : ι → ENat
    ⊢ Eq (HAdd.hAdd (iSup fun i => f i) a) (iSup fun i => HAdd.hAdd (f i) a)
  -/
  simp [add_comm, add_iSup]
  /-
    🎉 no goals
  -/


lemma add_biSup' {p : ι → Prop} (h : ∃ i, p i) (f : ι → ℕ∞) :
    a + ⨆ i, ⨆ _ : p i, f i = ⨆ i, ⨆ _ : p i, a + f i := by
  /-
    ι : Sort u_2
    a : ENat
    p : ι → Prop
    h : Exists fun i => p i
    f : ι → ENat
    ⊢ Eq (HAdd.hAdd a (iSup fun i => iSup fun x => f i)) (iSup fun i => iSup fun x …
  -/
  haveI : Nonempty {i // p i} := nonempty_subtype.2 h
  /-
    ι : Sort u_2
    a : ENat
    p : ι → Prop
    h : Exists fun i => p i
    f : ι → ENat
    this : Nonempty (Subtype fun i => p i)
    ⊢ Eq (HAdd.hAdd a (iSup fun i => iSup fun x => f i)) (iSup fun i => iSup fun x …
  -/
  simp only [iSup_subtype', add_iSup]
  /-
    🎉 no goals
  -/


lemma biSup_add' {p : ι → Prop} (h : ∃ i, p i) (f : ι → ℕ∞) :
                                                              /-
                                                                ι : Sort u_2
                                                                a : ENat
                                                                p : ι → Prop
                                                                h : Exists fun i => p i
                                                                f : ι → ENat
                                                                ⊢ Eq (HAdd.hAdd (iSup fun i => iSup fun x => f i) a) (iSup fun i => iSup fun x …
                                                              -/
    (⨆ i, ⨆ _ : p i, f i) + a = ⨆ i, ⨆ _ : p i, f i + a := by simp only [add_comm, add_biSup' h]
                                                              /-
                                                                🎉 no goals
                                                              -/


lemma add_biSup {ι : Type*} {s : Set ι} (hs : s.Nonempty) (f : ι → ℕ∞) :
    a + ⨆ i ∈ s, f i = ⨆ i ∈ s, a + f i := add_biSup' hs _


lemma biSup_add {ι : Type*} {s : Set ι} (hs : s.Nonempty) (f : ι → ℕ∞) :
    (⨆ i ∈ s, f i) + a = ⨆ i ∈ s, f i + a := biSup_add' hs _


lemma add_sSup (hs : s.Nonempty) : a + sSup s = ⨆ b ∈ s, a + b := by
  /-
    s : Set ENat
    a : ENat
    hs : s.Nonempty
    ⊢ Eq (HAdd.hAdd a (SupSet.sSup s)) (iSup fun b => iSup fun h => HAdd.hAdd a b)
  -/
  rw [sSup_eq_iSup, add_biSup hs]
  /-
    🎉 no goals
  -/


lemma sSup_add (hs : s.Nonempty) : sSup s + a = ⨆ b ∈ s, b + a := by
  /-
    s : Set ENat
    a : ENat
    hs : s.Nonempty
    ⊢ Eq (HAdd.hAdd (SupSet.sSup s) a) (iSup fun b => iSup fun h => HAdd.hAdd b a)
  -/
  rw [sSup_eq_iSup, biSup_add hs]
  /-
    🎉 no goals
  -/


lemma iSup_add_iSup_le [Nonempty ι] [Nonempty κ] {g : κ → ℕ∞} (h : ∀ i j, f i + g j ≤ a) :
                              /-
                                ι : Sort u_2
                                κ : Sort u_3
                                f : ι → ENat
                                a : ENat
                                inst✝¹ : Nonempty ι
                                inst✝ : Nonempty κ
                                g : κ → ENat
                                h : ∀ (i : ι) (j : κ), LE.le (HAdd.hAdd (f i) (g j)) a
                                ⊢ LE.le (HAdd.hAdd (iSup f) (iSup g)) a
                              -/
    iSup f + iSup g ≤ a := by simp_rw [iSup_add, add_iSup]; exact iSup₂_le h
                                                            /-
                                                              🎉 no goals
                                                            -/


lemma biSup_add_biSup_le' {p : ι → Prop} {q : κ → Prop} (hp : ∃ i, p i) (hq : ∃ j, q j)
    {g : κ → ℕ∞} (h : ∀ i, p i → ∀ j, q j → f i + g j ≤ a) :
    (⨆ i, ⨆ _ : p i, f i) + ⨆ j, ⨆ _ : q j, g j ≤ a := by
  /-
    ι : Sort u_2
    κ : Sort u_3
    f : ι → ENat
    a : ENat
    p : ι → Prop
    q : κ → Prop
    hp : Exists fun i => p i
    hq : Exists fun j => q j
    g : κ → ENat
    h : ∀ (i : ι), p i → ∀ (j : κ), q j → LE.le (HAdd.hAdd (f i) (g j)) a
    ⊢ LE.le (HAdd.hAdd (iSup fun i => iSup fun x => f i) (iSup fun j => iSup fun x …
  -/
  simp_rw [biSup_add' hp, add_biSup' hq]
  /-
    ι : Sort u_2
    κ : Sort u_3
    f : ι → ENat
    a : ENat
    p : ι → Prop
    q : κ → Prop
    hp : Exists fun i => p i
    hq : Exists fun j => q j
    g : κ → ENat
    h : ∀ (i : ι), p i → ∀ (j : κ), q j → LE.le (HAdd.hAdd (f i) (g j)) a
    ⊢ LE.le (iSup fun i => iSup fun x => iSup fun i_1 => iSup fun x => HAdd.hAdd ( …
  -/
  exact iSup₂_le fun i hi => iSup₂_le (h i hi)
  /-
    🎉 no goals
  -/


lemma biSup_add_biSup_le {ι κ : Type*} {s : Set ι} {t : Set κ} (hs : s.Nonempty) (ht : t.Nonempty)
    {f : ι → ℕ∞} {g : κ → ℕ∞} {a : ℕ∞} (h : ∀ i ∈ s, ∀ j ∈ t, f i + g j ≤ a) :
    (⨆ i ∈ s, f i) + ⨆ j ∈ t, g j ≤ a := biSup_add_biSup_le' hs ht h


lemma iSup_add_iSup (h : ∀ i j, ∃ k, f i + g j ≤ f k + g k) : iSup f + iSup g = ⨆ i, f i + g i := by
  /-
    ι : Sort u_2
    f g : ι → ENat
    h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
    ⊢ Eq (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
  -/
  cases isEmpty_or_nonempty ι
    /-
      case inl
      ι : Sort u_2
      f g : ι → ENat
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : IsEmpty ι
      ⊢ Eq (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
  · simp only [iSup_of_empty, bot_eq_zero, zero_add]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Sort u_2
      f g : ι → ENat
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      ⊢ Eq (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
  · refine le_antisymm ?_ (iSup_le fun a => add_le_add (le_iSup _ _) (le_iSup _ _))
    /-
      case inr
      ι : Sort u_2
      f g : ι → ENat
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      ⊢ LE.le (HAdd.hAdd (iSup f) (iSup g)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
    refine iSup_add_iSup_le fun i j => ?_
    /-
      case inr
      ι : Sort u_2
      f g : ι → ENat
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      i j : ι
      ⊢ LE.le (HAdd.hAdd (f i) (g j)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
    rcases h i j with ⟨k, hk⟩
    /-
      case inr.intro
      ι : Sort u_2
      f g : ι → ENat
      h : ∀ (i j : ι), Exists fun k => LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k …
      h✝ : Nonempty ι
      i j k : ι
      hk : LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f k) (g k))
      ⊢ LE.le (HAdd.hAdd (f i) (g j)) (iSup fun i => HAdd.hAdd (f i) (g i))
    -/
    exact le_iSup_of_le k hk
    /-
      🎉 no goals
    -/


lemma iSup_add_iSup_of_monotone {ι : Type*} [Preorder ι] [IsDirected ι (· ≤ ·)] {f g : ι → ℕ∞}
    (hf : Monotone f) (hg : Monotone g) : iSup f + iSup g = ⨆ a, f a + g a :=
                                                                      /-
                                                                        ι : Type u_4
                                                                        inst✝¹ : Preorder ι
                                                                        inst✝ : IsDirected ι fun x1 x2 => LE.le x1 x2
                                                                        f g : ι → ENat
                                                                        hf : Monotone f
                                                                        hg : Monotone g
                                                                        i j _k : ι
                                                                        x✝ : And (LE.le i _k) (LE.le j _k)
                                                                        hi : LE.le i _k
                                                                        hj : LE.le j _k
                                                                        ⊢ LE.le (HAdd.hAdd (f i) (g j)) (HAdd.hAdd (f _k) (g _k))
                                                                      -/
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  iSup_add_iSup fun i j ↦ (exists_ge_ge i j).imp fun _k ⟨hi, hj⟩ ↦ by gcongr <;> apply_rules
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


lemma sub_iSup [Nonempty ι] (ha : a ≠ ⊤) : a - ⨆ i, f i = ⨅ i, a - f i := by
  /-
    ι : Sort u_2
    f : ι → ENat
    a : ENat
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    ⊢ Eq (HSub.hSub a (iSup fun i => f i)) (iInf fun i => HSub.hSub a (f i))
  -/
  obtain ⟨i, hi⟩ | h := em (∃ i, a < f i)
    /-
      case inl.intro
      ι : Sort u_2
      f : ι → ENat
      a : ENat
      inst✝ : Nonempty ι
      ha : Ne a Top.top
      i : ι
      hi : LT.lt a (f i)
      ⊢ Eq (HSub.hSub a (iSup fun i => f i)) (iInf fun i => HSub.hSub a (f i))
    -/
  · rw [tsub_eq_zero_iff_le.2 <| le_iSup_of_le _ hi.le, (iInf_eq_bot _).2, bot_eq_zero]
    /-
      case inl.intro
      ι : Sort u_2
      f : ι → ENat
      a : ENat
      inst✝ : Nonempty ι
      ha : Ne a Top.top
      i : ι
      hi : LT.lt a (f i)
      ⊢ ∀ (b : ENat), GT.gt b Bot.bot → Exists fun i => LT.lt (HSub.hSub a (f i)) b
    -/
    exact fun x hx ↦ ⟨i, by simpa [hi.le, tsub_eq_zero_of_le]⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Sort u_2
    f : ι → ENat
    a : ENat
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    h : Not (Exists fun i => LT.lt a (f i))
    ⊢ Eq (HSub.hSub a (iSup fun i => f i)) (iInf fun i => HSub.hSub a (f i))
  -/
  simp_rw [not_exists, not_lt] at h
  refine le_antisymm (le_iInf fun i ↦ tsub_le_tsub_left (le_iSup ..) _) <|
    ENat.le_sub_of_add_le_left (ne_top_of_le_ne_top ha <| iSup_le h) <|
    add_le_of_le_tsub_right_of_le (iInf_le_of_le (Classical.arbitrary _) tsub_le_self) <|
    iSup_le fun i ↦ ?_
  /-
    case inr
    ι : Sort u_2
    f : ι → ENat
    a : ENat
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    h : ∀ (x : ι), LE.le (f x) a
    i : ι
    ⊢ LE.le (f i) (HSub.hSub a (iInf fun i => HSub.hSub a (f i)))
  -/
  rw [← ENat.sub_sub_cancel ha (h _)]
  /-
    case inr
    ι : Sort u_2
    f : ι → ENat
    a : ENat
    inst✝ : Nonempty ι
    ha : Ne a Top.top
    h : ∀ (x : ι), LE.le (f x) a
    i : ι
    ⊢ LE.le (HSub.hSub a (HSub.hSub a (f i))) (HSub.hSub a (iInf fun i => HSub.hSu …
  -/
  exact tsub_le_tsub_left (iInf_le (a - f ·) i) _
  /-
    🎉 no goals
  -/


