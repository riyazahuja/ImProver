/-- If `α` is an infinite type, then `α × α` and `α` have the same cardinality. -/
theorem mul_eq_self {c : Cardinal} (h : ℵ₀ ≤ c) : c * c = c := by
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ Eq (HMul.hMul c c) c
  -/
  refine le_antisymm ?_ (by simpa only [mul_one] using mul_le_mul_left' (one_le_aleph0.trans h) c)
  -- the only nontrivial part is `c * c ≤ c`. We prove it inductively.
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ LE.le (HMul.hMul c c) c
  -/
  refine Acc.recOn (Cardinal.lt_wf.apply c) (fun c _ => Cardinal.inductionOn c fun α IH ol => ?_) h
  -- consider the minimal well-order `r` on `α` (a type with cardinality `c`).
  /-
    c✝ : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c✝
    c : Cardinal.{u_1}
    x✝ : ∀ (y : Cardinal.{u_1}), LT.lt y c → Acc (fun x1 x2 => LT.lt x1 x2) y
    α : Type u_1
    IH : ∀ (y : Cardinal.{u_1}), LT.lt y (Cardinal.mk α) → LE.le Cardinal.aleph0 y …
    ol : LE.le Cardinal.aleph0 (Cardinal.mk α)
    ⊢ LE.le (HMul.hMul (Cardinal.mk α) (Cardinal.mk α)) (Cardinal.mk α)
  -/
  rcases ord_eq α with ⟨r, wo, e⟩
  classical
  letI := linearOrderOfSTO r
  haveI : IsWellOrder α (· < ·) := wo
  -- Define an order `s` on `α × α` by writing `(a, b) < (c, d)` if `max a b < max c d`, or
  -- the max are equal and `a < c`, or the max are equal and `a = c` and `b < d`.
  let g : α × α → α := fun p => max p.1 p.2
  let f : α × α ↪ Ordinal × α × α :=
    ⟨fun p : α × α => (typein (· < ·) (g p), p), fun p q => congr_arg Prod.snd⟩
  let s := f ⁻¹'o Prod.Lex (· < ·) (Prod.Lex (· < ·) (· < ·))
  -- this is a well order on `α × α`.
  haveI : IsWellOrder _ s := (RelEmbedding.preimage _ _).isWellOrder
  /- it suffices to show that this well order is smaller than `r`
       if it were larger, then `r` would be a strict prefix of `s`. It would be contained in
      `β × β` for some `β` of cardinality `< c`. By the inductive assumption, this set has the
      same cardinality as `β` (or it is finite if `β` is finite), so it is `< c`, which is a
      contradiction. -/
  suffices type s ≤ type r by exact card_le_card this
  refine le_of_forall_lt fun o h => ?_
  rcases typein_surj s h with ⟨p, rfl⟩
  rw [← e, lt_ord]
  refine lt_of_le_of_lt
    (?_ : _ ≤ card (succ (typein (· < ·) (g p))) * card (succ (typein (· < ·) (g p)))) ?_
  · have : { q | s q p } ⊆ insert (g p) { x | x < g p } ×ˢ insert (g p) { x | x < g p } := by
      intro q h
      simp only [s, f, Preimage, Embedding.coeFn_mk, Prod.lex_def, typein_lt_typein,
        typein_inj, mem_setOf_eq] at h
      exact max_le_iff.1 (le_iff_lt_or_eq.2 <| h.imp_right And.left)
    suffices H : (insert (g p) { x | r x (g p) } : Set α) ≃ { x | r x (g p) } ⊕ PUnit from
      ⟨(Set.embeddingOfSubset _ _ this).trans
        ((Equiv.Set.prod _ _).trans (H.prodCongr H)).toEmbedding⟩
    refine (Equiv.Set.insert ?_).trans ((Equiv.refl _).sumCongr punitEquivPUnit)
    apply @irrefl _ r
  cases' lt_or_le (card (succ (typein (· < ·) (g p)))) ℵ₀ with qo qo
  · exact (mul_lt_aleph0 qo qo).trans_le ol
  · suffices (succ (typein LT.lt (g p))).card < #α from (IH _ this qo).trans_lt this
    rw [← lt_ord]
    apply (isLimit_ord ol).succ_lt
    rw [e]
    apply typein_lt_type


/-- If `α` and `β` are infinite types, then the cardinality of `α × β` is the maximum
of the cardinalities of `α` and `β`. -/
theorem mul_eq_max {a b : Cardinal} (ha : ℵ₀ ≤ a) (hb : ℵ₀ ≤ b) : a * b = max a b :=
  le_antisymm
      (mul_eq_self (ha.trans (le_max_left a b)) ▸
        mul_le_mul' (le_max_left _ _) (le_max_right _ _)) <|
               /-
                 a b : Cardinal.{u_1}
                 ha : LE.le Cardinal.aleph0 a
                 hb : LE.le Cardinal.aleph0 b
                 ⊢ LE.le a (HMul.hMul a b)
               -/
    max_le (by simpa only [mul_one] using mul_le_mul_left' (one_le_aleph0.trans hb) a)
               /-
                 🎉 no goals
               -/
          /-
            a b : Cardinal.{u_1}
            ha : LE.le Cardinal.aleph0 a
            hb : LE.le Cardinal.aleph0 b
            ⊢ LE.le b (HMul.hMul a b)
          -/
      (by simpa only [one_mul] using mul_le_mul_right' (one_le_aleph0.trans ha) b)
          /-
            🎉 no goals
          -/


@[simp]
theorem mul_mk_eq_max {α β : Type u} [Infinite α] [Infinite β] : #α * #β = max #α #β :=
  mul_eq_max (aleph0_le_mk α) (aleph0_le_mk β)


@[simp]
theorem aleph_mul_aleph (o₁ o₂ : Ordinal) : ℵ_ o₁ * ℵ_ o₂ = ℵ_ (max o₁ o₂) := by
  /-
    o₁ o₂ : Ordinal.{u_1}
    ⊢ Eq (HMul.hMul (Cardinal.aleph o₁) (Cardinal.aleph o₂)) (Cardinal.aleph (Max. …
  -/
  rw [Cardinal.mul_eq_max (aleph0_le_aleph o₁) (aleph0_le_aleph o₂), aleph_max]
  /-
    🎉 no goals
  -/


@[simp]
theorem aleph0_mul_eq {a : Cardinal} (ha : ℵ₀ ≤ a) : ℵ₀ * a = a :=
  (mul_eq_max le_rfl ha).trans (max_eq_right ha)


@[simp]
theorem mul_aleph0_eq {a : Cardinal} (ha : ℵ₀ ≤ a) : a * ℵ₀ = a :=
  (mul_eq_max ha le_rfl).trans (max_eq_left ha)


theorem aleph0_mul_mk_eq {α : Type*} [Infinite α] : ℵ₀ * #α = #α :=
  aleph0_mul_eq (aleph0_le_mk α)


theorem mk_mul_aleph0_eq {α : Type*} [Infinite α] : #α * ℵ₀ = #α :=
  mul_aleph0_eq (aleph0_le_mk α)


@[simp]
theorem aleph0_mul_aleph (o : Ordinal) : ℵ₀ * ℵ_ o = ℵ_ o :=
  aleph0_mul_eq (aleph0_le_aleph o)


@[simp]
theorem aleph_mul_aleph0 (o : Ordinal) : ℵ_ o * ℵ₀ = ℵ_ o :=
  mul_aleph0_eq (aleph0_le_aleph o)


theorem mul_lt_of_lt {a b c : Cardinal} (hc : ℵ₀ ≤ c) (h1 : a < c) (h2 : b < c) : a * b < c :=
  (mul_le_mul' (le_max_left a b) (le_max_right a b)).trans_lt <|
    (lt_or_le (max a b) ℵ₀).elim (fun h => (mul_lt_aleph0 h h).trans_le hc) fun h => by
      /-
        a b c : Cardinal.{u_1}
        hc : LE.le Cardinal.aleph0 c
        h1 : LT.lt a c
        h2 : LT.lt b c
        h : LE.le Cardinal.aleph0 (Max.max a b)
        ⊢ LT.lt (HMul.hMul (Max.max a b) (Max.max a b)) c
      -/
      rw [mul_eq_self h]
      /-
        a b c : Cardinal.{u_1}
        hc : LE.le Cardinal.aleph0 c
        h1 : LT.lt a c
        h2 : LT.lt b c
        h : LE.le Cardinal.aleph0 (Max.max a b)
        ⊢ LT.lt (Max.max a b) c
      -/
      exact max_lt h1 h2
      /-
        🎉 no goals
      -/


theorem mul_le_max_of_aleph0_le_left {a b : Cardinal} (h : ℵ₀ ≤ a) : a * b ≤ max a b := by
  /-
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    ⊢ LE.le (HMul.hMul a b) (Max.max a b)
  -/
  convert mul_le_mul' (le_max_left a b) (le_max_right a b) using 1
  /-
    case h.e'_4
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    ⊢ Eq (Max.max a b) (HMul.hMul (Max.max a b) (Max.max a b))
  -/
  rw [mul_eq_self]
  /-
    case h.e'_4
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    ⊢ LE.le Cardinal.aleph0 (Max.max a b)
  -/
  exact h.trans (le_max_left a b)
  /-
    🎉 no goals
  -/


theorem mul_eq_max_of_aleph0_le_left {a b : Cardinal} (h : ℵ₀ ≤ a) (h' : b ≠ 0) :
    a * b = max a b := by
  /-
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    h' : Ne b 0
    ⊢ Eq (HMul.hMul a b) (Max.max a b)
  -/
  rcases le_or_lt ℵ₀ b with hb | hb
    /-
      case inl
      a b : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 a
      h' : Ne b 0
      hb : LE.le Cardinal.aleph0 b
      ⊢ Eq (HMul.hMul a b) (Max.max a b)
    -/
  · exact mul_eq_max h hb
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    h' : Ne b 0
    hb : LT.lt b Cardinal.aleph0
    ⊢ Eq (HMul.hMul a b) (Max.max a b)
  -/
  refine (mul_le_max_of_aleph0_le_left h).antisymm ?_
  /-
    case inr
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    h' : Ne b 0
    hb : LT.lt b Cardinal.aleph0
    ⊢ LE.le (Max.max a b) (HMul.hMul a b)
  -/
  have : b ≤ a := hb.le.trans h
  /-
    case inr
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    h' : Ne b 0
    hb : LT.lt b Cardinal.aleph0
    this : LE.le b a
    ⊢ LE.le (Max.max a b) (HMul.hMul a b)
  -/
  rw [max_eq_left this]
  /-
    case inr
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    h' : Ne b 0
    hb : LT.lt b Cardinal.aleph0
    this : LE.le b a
    ⊢ LE.le a (HMul.hMul a b)
  -/
  convert mul_le_mul_left' (one_le_iff_ne_zero.mpr h') a
  /-
    case h.e'_3
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 a
    h' : Ne b 0
    hb : LT.lt b Cardinal.aleph0
    this : LE.le b a
    ⊢ Eq a (HMul.hMul a 1)
  -/
  rw [mul_one]
  /-
    🎉 no goals
  -/


theorem mul_le_max_of_aleph0_le_right {a b : Cardinal} (h : ℵ₀ ≤ b) : a * b ≤ max a b := by
  /-
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 b
    ⊢ LE.le (HMul.hMul a b) (Max.max a b)
  -/
  simpa only [mul_comm b, max_comm b] using mul_le_max_of_aleph0_le_left h
  /-
    🎉 no goals
  -/


theorem mul_eq_max_of_aleph0_le_right {a b : Cardinal} (h' : a ≠ 0) (h : ℵ₀ ≤ b) :
    a * b = max a b := by
  /-
    a b : Cardinal.{u_1}
    h' : Ne a 0
    h : LE.le Cardinal.aleph0 b
    ⊢ Eq (HMul.hMul a b) (Max.max a b)
  -/
  rw [mul_comm, max_comm]
  /-
    a b : Cardinal.{u_1}
    h' : Ne a 0
    h : LE.le Cardinal.aleph0 b
    ⊢ Eq (HMul.hMul b a) (Max.max b a)
  -/
  exact mul_eq_max_of_aleph0_le_left h h'
  /-
    🎉 no goals
  -/


theorem mul_eq_max' {a b : Cardinal} (h : ℵ₀ ≤ a * b) : a * b = max a b := by
  /-
    a b : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 (HMul.hMul a b)
    ⊢ Eq (HMul.hMul a b) (Max.max a b)
  -/
  rcases aleph0_le_mul_iff.mp h with ⟨ha, hb, ha' | hb'⟩
    /-
      case intro.intro.inl
      a b : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 (HMul.hMul a b)
      ha : Ne a 0
      hb : Ne b 0
      ha' : LE.le Cardinal.aleph0 a
      ⊢ Eq (HMul.hMul a b) (Max.max a b)
    -/
  · exact mul_eq_max_of_aleph0_le_left ha' hb
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.inr
      a b : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 (HMul.hMul a b)
      ha : Ne a 0
      hb : Ne b 0
      hb' : LE.le Cardinal.aleph0 b
      ⊢ Eq (HMul.hMul a b) (Max.max a b)
    -/
  · exact mul_eq_max_of_aleph0_le_right ha hb'
    /-
      🎉 no goals
    -/


theorem mul_le_max (a b : Cardinal) : a * b ≤ max (max a b) ℵ₀ := by
  /-
    a b : Cardinal.{u_1}
    ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
  -/
  rcases eq_or_ne a 0 with (rfl | ha0); · simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr
    a b : Cardinal.{u_1}
    ha0 : Ne a 0
    ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
  -/
  rcases eq_or_ne b 0 with (rfl | hb0); · simp
                                          /-
                                            🎉 no goals
                                          -/
  /-
    case inr.inr
    a b : Cardinal.{u_1}
    ha0 : Ne a 0
    hb0 : Ne b 0
    ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
  -/
  rcases le_or_lt ℵ₀ a with ha | ha
    /-
      case inr.inr.inl
      a b : Cardinal.{u_1}
      ha0 : Ne a 0
      hb0 : Ne b 0
      ha : LE.le Cardinal.aleph0 a
      ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
    -/
  · rw [mul_eq_max_of_aleph0_le_left ha hb0]
    /-
      case inr.inr.inl
      a b : Cardinal.{u_1}
      ha0 : Ne a 0
      hb0 : Ne b 0
      ha : LE.le Cardinal.aleph0 a
      ⊢ LE.le (Max.max a b) (Max.max (Max.max a b) Cardinal.aleph0)
    -/
    exact le_max_left _ _
    /-
      🎉 no goals
    -/
    /-
      case inr.inr.inr
      a b : Cardinal.{u_1}
      ha0 : Ne a 0
      hb0 : Ne b 0
      ha : LT.lt a Cardinal.aleph0
      ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
    -/
  · rcases le_or_lt ℵ₀ b with hb | hb
      /-
        case inr.inr.inr.inl
        a b : Cardinal.{u_1}
        ha0 : Ne a 0
        hb0 : Ne b 0
        ha : LT.lt a Cardinal.aleph0
        hb : LE.le Cardinal.aleph0 b
        ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
      -/
    · rw [mul_comm, mul_eq_max_of_aleph0_le_left hb ha0, max_comm]
      /-
        case inr.inr.inr.inl
        a b : Cardinal.{u_1}
        ha0 : Ne a 0
        hb0 : Ne b 0
        ha : LT.lt a Cardinal.aleph0
        hb : LE.le Cardinal.aleph0 b
        ⊢ LE.le (Max.max a b) (Max.max (Max.max a b) Cardinal.aleph0)
      -/
      exact le_max_left _ _
      /-
        🎉 no goals
      -/
      /-
        case inr.inr.inr.inr
        a b : Cardinal.{u_1}
        ha0 : Ne a 0
        hb0 : Ne b 0
        ha : LT.lt a Cardinal.aleph0
        hb : LT.lt b Cardinal.aleph0
        ⊢ LE.le (HMul.hMul a b) (Max.max (Max.max a b) Cardinal.aleph0)
      -/
    · exact le_max_of_le_right (mul_lt_aleph0 ha hb).le
      /-
        🎉 no goals
      -/


theorem mul_eq_left {a b : Cardinal} (ha : ℵ₀ ≤ a) (hb : b ≤ a) (hb' : b ≠ 0) : a * b = a := by
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 a
    hb : LE.le b a
    hb' : Ne b 0
    ⊢ Eq (HMul.hMul a b) a
  -/
  rw [mul_eq_max_of_aleph0_le_left ha hb', max_eq_left hb]
  /-
    🎉 no goals
  -/


theorem mul_eq_right {a b : Cardinal} (hb : ℵ₀ ≤ b) (ha : a ≤ b) (ha' : a ≠ 0) : a * b = b := by
  /-
    a b : Cardinal.{u_1}
    hb : LE.le Cardinal.aleph0 b
    ha : LE.le a b
    ha' : Ne a 0
    ⊢ Eq (HMul.hMul a b) b
  -/
  rw [mul_comm, mul_eq_left hb ha ha']
  /-
    🎉 no goals
  -/


theorem le_mul_left {a b : Cardinal} (h : b ≠ 0) : a ≤ b * a := by
  /-
    a b : Cardinal.{u_1}
    h : Ne b 0
    ⊢ LE.le a (HMul.hMul b a)
  -/
  convert mul_le_mul_right' (one_le_iff_ne_zero.mpr h) a
  /-
    case h.e'_3
    a b : Cardinal.{u_1}
    h : Ne b 0
    ⊢ Eq a (HMul.hMul 1 a)
  -/
  rw [one_mul]
  /-
    🎉 no goals
  -/


theorem le_mul_right {a b : Cardinal} (h : b ≠ 0) : a ≤ a * b := by
  /-
    a b : Cardinal.{u_1}
    h : Ne b 0
    ⊢ LE.le a (HMul.hMul a b)
  -/
  rw [mul_comm]
  /-
    a b : Cardinal.{u_1}
    h : Ne b 0
    ⊢ LE.le a (HMul.hMul b a)
  -/
  exact le_mul_left h
  /-
    🎉 no goals
  -/


theorem mul_eq_left_iff {a b : Cardinal} : a * b = a ↔ max ℵ₀ b ≤ a ∧ b ≠ 0 ∨ b = 1 ∨ a = 0 := by
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (Eq (HMul.hMul a b) a) (Or (And (LE.le (Max.max Cardinal.aleph0 b) a) (N …
  -/
  rw [max_le_iff]
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (Eq (HMul.hMul a b) a) (Or (And (And (LE.le Cardinal.aleph0 a) (LE.le b  …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      a b : Cardinal.{u_1}
      h : Eq (HMul.hMul a b) a
      ⊢ Or (And (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Ne b 0)) (Or (Eq b 1) ( …
    -/
  · rcases le_or_lt ℵ₀ a with ha | ha
    · have : a ≠ 0 := by
        rintro rfl
        exact ha.not_lt aleph0_pos
      /-
        case refine_1.inl
        a b : Cardinal.{u_1}
        h : Eq (HMul.hMul a b) a
        ha : LE.le Cardinal.aleph0 a
        this : Ne a 0
        ⊢ Or (And (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Ne b 0)) (Or (Eq b 1) ( …
      -/
      left
      /-
        case refine_1.inl.h
        a b : Cardinal.{u_1}
        h : Eq (HMul.hMul a b) a
        ha : LE.le Cardinal.aleph0 a
        this : Ne a 0
        ⊢ And (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Ne b 0)
      -/
      rw [and_assoc]
      /-
        case refine_1.inl.h
        a b : Cardinal.{u_1}
        h : Eq (HMul.hMul a b) a
        ha : LE.le Cardinal.aleph0 a
        this : Ne a 0
        ⊢ And (LE.le Cardinal.aleph0 a) (And (LE.le b a) (Ne b 0))
      -/
      use ha
      /-
        case right
        a b : Cardinal.{u_1}
        h : Eq (HMul.hMul a b) a
        ha : LE.le Cardinal.aleph0 a
        this : Ne a 0
        ⊢ And (LE.le b a) (Ne b 0)
      -/
      constructor
        /-
          case right.left
          a b : Cardinal.{u_1}
          h : Eq (HMul.hMul a b) a
          ha : LE.le Cardinal.aleph0 a
          this : Ne a 0
          ⊢ LE.le b a
        -/
      · rw [← not_lt]
        /-
          case right.left
          a b : Cardinal.{u_1}
          h : Eq (HMul.hMul a b) a
          ha : LE.le Cardinal.aleph0 a
          this : Ne a 0
          ⊢ Not (LT.lt a b)
        -/
        exact fun hb => ne_of_gt (hb.trans_le (le_mul_left this)) h
        /-
          🎉 no goals
        -/
        /-
          case right.right
          a b : Cardinal.{u_1}
          h : Eq (HMul.hMul a b) a
          ha : LE.le Cardinal.aleph0 a
          this : Ne a 0
          ⊢ Ne b 0
        -/
      · rintro rfl
        /-
          case right.right
          a : Cardinal.{u_1}
          ha : LE.le Cardinal.aleph0 a
          this : Ne a 0
          h : Eq (HMul.hMul a 0) a
          ⊢ False
        -/
        apply this
        /-
          case right.right
          a : Cardinal.{u_1}
          ha : LE.le Cardinal.aleph0 a
          this : Ne a 0
          h : Eq (HMul.hMul a 0) a
          ⊢ Eq a 0
        -/
        rw [mul_zero] at h
        /-
          case right.right
          a : Cardinal.{u_1}
          ha : LE.le Cardinal.aleph0 a
          this : Ne a 0
          h : Eq 0 a
          ⊢ Eq a 0
        -/
        exact h.symm
        /-
          🎉 no goals
        -/
    /-
      case refine_1.inr
      a b : Cardinal.{u_1}
      h : Eq (HMul.hMul a b) a
      ha : LT.lt a Cardinal.aleph0
      ⊢ Or (And (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Ne b 0)) (Or (Eq b 1) ( …
    -/
    right
    /-
      case refine_1.inr.h
      a b : Cardinal.{u_1}
      h : Eq (HMul.hMul a b) a
      ha : LT.lt a Cardinal.aleph0
      ⊢ Or (Eq b 1) (Eq a 0)
    -/
    by_cases h2a : a = 0
      /-
        case pos
        a b : Cardinal.{u_1}
        h : Eq (HMul.hMul a b) a
        ha : LT.lt a Cardinal.aleph0
        h2a : Eq a 0
        ⊢ Or (Eq b 1) (Eq a 0)
      -/
    · exact Or.inr h2a
      /-
        🎉 no goals
      -/
    have hb : b ≠ 0 := by
      rintro rfl
      apply h2a
      rw [mul_zero] at h
      exact h.symm
    /-
      case neg
      a b : Cardinal.{u_1}
      h : Eq (HMul.hMul a b) a
      ha : LT.lt a Cardinal.aleph0
      h2a : Not (Eq a 0)
      hb : Ne b 0
      ⊢ Or (Eq b 1) (Eq a 0)
    -/
    left
    /-
      case neg.h
      a b : Cardinal.{u_1}
      h : Eq (HMul.hMul a b) a
      ha : LT.lt a Cardinal.aleph0
      h2a : Not (Eq a 0)
      hb : Ne b 0
      ⊢ Eq b 1
    -/
    rw [← h, mul_lt_aleph0_iff, lt_aleph0, lt_aleph0] at ha
    /-
      case neg.h
      a b : Cardinal.{u_1}
      h : Eq (HMul.hMul a b) a
      ha : Or (Eq a 0) (Or (Eq b 0) (And (Exists fun n => Eq a ↑n) (Exists fun n =>  …
      h2a : Not (Eq a 0)
      hb : Ne b 0
      ⊢ Eq b 1
    -/
    rcases ha with (rfl | rfl | ⟨⟨n, rfl⟩, ⟨m, rfl⟩⟩)
      /-
        case neg.h.inl
        b : Cardinal.{u_1}
        hb : Ne b 0
        h : Eq (HMul.hMul 0 b) 0
        h2a : Not (Eq 0 0)
        ⊢ Eq b 1
      -/
    · contradiction
      /-
        🎉 no goals
      -/
      /-
        case neg.h.inr.inl
        a : Cardinal.{u_1}
        h2a : Not (Eq a 0)
        h : Eq (HMul.hMul a 0) a
        hb : Ne 0 0
        ⊢ Eq 0 1
      -/
    · contradiction
      /-
        🎉 no goals
      -/
    /-
      case neg.h.inr.inr.intro.intro.intro
      n : Nat
      h2a : Not (Eq (↑n) 0)
      m : Nat
      hb : Ne (↑m) 0
      h : Eq (HMul.hMul ↑n ↑m) ↑n
      ⊢ Eq (↑m) 1
    -/
    rw [← Ne] at h2a
    /-
      case neg.h.inr.inr.intro.intro.intro
      n : Nat
      h2a : Ne (↑n) 0
      m : Nat
      hb : Ne (↑m) 0
      h : Eq (HMul.hMul ↑n ↑m) ↑n
      ⊢ Eq (↑m) 1
    -/
    rw [← one_le_iff_ne_zero] at h2a hb
    /-
      case neg.h.inr.inr.intro.intro.intro
      n : Nat
      h2a : LE.le 1 ↑n
      m : Nat
      hb : LE.le 1 ↑m
      h : Eq (HMul.hMul ↑n ↑m) ↑n
      ⊢ Eq (↑m) 1
    -/
    norm_cast at h2a hb h ⊢
    /-
      case neg.h.inr.inr.intro.intro.intro
      n m : Nat
      h2a : LE.le 1 n
      hb : LE.le 1 m
      h : Eq (HMul.hMul n m) n
      ⊢ Eq m 1
    -/
    apply le_antisymm _ hb
    /-
      n m : Nat
      h2a : LE.le 1 n
      hb : LE.le 1 m
      h : Eq (HMul.hMul n m) n
      ⊢ LE.le m 1
    -/
    rw [← not_lt]
    /-
      n m : Nat
      h2a : LE.le 1 n
      hb : LE.le 1 m
      h : Eq (HMul.hMul n m) n
      ⊢ Not (LT.lt 1 m)
    -/
    apply fun h2b => ne_of_gt _ h
    /-
      n m : Nat
      h2a : LE.le 1 n
      hb : LE.le 1 m
      h : Eq (HMul.hMul n m) n
      ⊢ LT.lt 1 m → LT.lt n (HMul.hMul n m)
    -/
    conv_rhs => left; rw [← mul_one n]
    /-
      n m : Nat
      h2a : LE.le 1 n
      hb : LE.le 1 m
      h : Eq (HMul.hMul n m) n
      ⊢ LT.lt 1 m → LT.lt (HMul.hMul n 1) (HMul.hMul n m)
    -/
    rw [mul_lt_mul_left]
      /-
        n m : Nat
        h2a : LE.le 1 n
        hb : LE.le 1 m
        h : Eq (HMul.hMul n m) n
        ⊢ LT.lt 1 m → LT.lt 1 m
      -/
    · exact id
      /-
        🎉 no goals
      -/
    /-
      n m : Nat
      h2a : LE.le 1 n
      hb : LE.le 1 m
      h : Eq (HMul.hMul n m) n
      ⊢ LT.lt 0 n
    -/
    apply Nat.lt_of_succ_le h2a
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Cardinal.{u_1}
      ⊢ Or (And (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Ne b 0)) (Or (Eq b 1) ( …
    -/
  · rintro (⟨⟨ha, hab⟩, hb⟩ | rfl | rfl)
      /-
        case refine_2.inl.intro.intro
        a b : Cardinal.{u_1}
        hb : Ne b 0
        ha : LE.le Cardinal.aleph0 a
        hab : LE.le b a
        ⊢ Eq (HMul.hMul a b) a
      -/
    · rw [mul_eq_max_of_aleph0_le_left ha hb, max_eq_left hab]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.inr.inl
      a : Cardinal.{u_1}
      ⊢ Eq (HMul.hMul a 1) a
    -/
    all_goals simp
    /-
      🎉 no goals
    -/


/-- If `α` is an infinite type, then `α ⊕ α` and `α` have the same cardinality. -/
theorem add_eq_self {c : Cardinal} (h : ℵ₀ ≤ c) : c + c = c :=
  le_antisymm
    (by
      /-
        c : Cardinal.{u_1}
        h : LE.le Cardinal.aleph0 c
        ⊢ LE.le (HAdd.hAdd c c) c
      -/
      convert mul_le_mul_right' ((nat_lt_aleph0 2).le.trans h) c using 1
          /-
            case h.e'_3
            c : Cardinal.{u_1}
            h : LE.le Cardinal.aleph0 c
            ⊢ Eq (HAdd.hAdd c c) (HMul.hMul (↑2) c)
          -/
          /-
            🎉 no goals
          -/
      <;> simp [two_mul, mul_eq_self h])
          /-
            🎉 no goals
          -/
    (self_le_add_left c c)


/-- If `α` is an infinite type, then the cardinality of `α ⊕ β` is the maximum
of the cardinalities of `α` and `β`. -/
theorem add_eq_max {a b : Cardinal} (ha : ℵ₀ ≤ a) : a + b = max a b :=
  le_antisymm
      (add_eq_self (ha.trans (le_max_left a b)) ▸
        add_le_add (le_max_left _ _) (le_max_right _ _)) <|
    max_le (self_le_add_right _ _) (self_le_add_left _ _)


theorem add_eq_max' {a b : Cardinal} (ha : ℵ₀ ≤ b) : a + b = max a b := by
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 b
    ⊢ Eq (HAdd.hAdd a b) (Max.max a b)
  -/
  rw [add_comm, max_comm, add_eq_max ha]
  /-
    🎉 no goals
  -/


@[simp]
theorem add_mk_eq_max {α β : Type u} [Infinite α] : #α + #β = max #α #β :=
  add_eq_max (aleph0_le_mk α)


@[simp]
theorem add_mk_eq_max' {α β : Type u} [Infinite β] : #α + #β = max #α #β :=
  add_eq_max' (aleph0_le_mk β)


theorem add_le_max (a b : Cardinal) : a + b ≤ max (max a b) ℵ₀ := by
  /-
    a b : Cardinal.{u_1}
    ⊢ LE.le (HAdd.hAdd a b) (Max.max (Max.max a b) Cardinal.aleph0)
  -/
  rcases le_or_lt ℵ₀ a with ha | ha
    /-
      case inl
      a b : Cardinal.{u_1}
      ha : LE.le Cardinal.aleph0 a
      ⊢ LE.le (HAdd.hAdd a b) (Max.max (Max.max a b) Cardinal.aleph0)
    -/
  · rw [add_eq_max ha]
    /-
      case inl
      a b : Cardinal.{u_1}
      ha : LE.le Cardinal.aleph0 a
      ⊢ LE.le (Max.max a b) (Max.max (Max.max a b) Cardinal.aleph0)
    -/
    exact le_max_left _ _
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Cardinal.{u_1}
      ha : LT.lt a Cardinal.aleph0
      ⊢ LE.le (HAdd.hAdd a b) (Max.max (Max.max a b) Cardinal.aleph0)
    -/
  · rcases le_or_lt ℵ₀ b with hb | hb
      /-
        case inr.inl
        a b : Cardinal.{u_1}
        ha : LT.lt a Cardinal.aleph0
        hb : LE.le Cardinal.aleph0 b
        ⊢ LE.le (HAdd.hAdd a b) (Max.max (Max.max a b) Cardinal.aleph0)
      -/
    · rw [add_comm, add_eq_max hb, max_comm]
      /-
        case inr.inl
        a b : Cardinal.{u_1}
        ha : LT.lt a Cardinal.aleph0
        hb : LE.le Cardinal.aleph0 b
        ⊢ LE.le (Max.max a b) (Max.max (Max.max a b) Cardinal.aleph0)
      -/
      exact le_max_left _ _
      /-
        🎉 no goals
      -/
      /-
        case inr.inr
        a b : Cardinal.{u_1}
        ha : LT.lt a Cardinal.aleph0
        hb : LT.lt b Cardinal.aleph0
        ⊢ LE.le (HAdd.hAdd a b) (Max.max (Max.max a b) Cardinal.aleph0)
      -/
    · exact le_max_of_le_right (add_lt_aleph0 ha hb).le
      /-
        🎉 no goals
      -/


theorem add_le_of_le {a b c : Cardinal} (hc : ℵ₀ ≤ c) (h1 : a ≤ c) (h2 : b ≤ c) : a + b ≤ c :=
  (add_le_add h1 h2).trans <| le_of_eq <| add_eq_self hc


theorem add_lt_of_lt {a b c : Cardinal} (hc : ℵ₀ ≤ c) (h1 : a < c) (h2 : b < c) : a + b < c :=
  (add_le_add (le_max_left a b) (le_max_right a b)).trans_lt <|
    (lt_or_le (max a b) ℵ₀).elim (fun h => (add_lt_aleph0 h h).trans_le hc) fun h => by
      /-
        a b c : Cardinal.{u_1}
        hc : LE.le Cardinal.aleph0 c
        h1 : LT.lt a c
        h2 : LT.lt b c
        h : LE.le Cardinal.aleph0 (Max.max a b)
        ⊢ LT.lt (HAdd.hAdd (Max.max a b) (Max.max a b)) c
      -/
      rw [add_eq_self h]; exact max_lt h1 h2
                          /-
                            🎉 no goals
                          -/


theorem eq_of_add_eq_of_aleph0_le {a b c : Cardinal} (h : a + b = c) (ha : a < c) (hc : ℵ₀ ≤ c) :
    b = c := by
  /-
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd a b) c
    ha : LT.lt a c
    hc : LE.le Cardinal.aleph0 c
    ⊢ Eq b c
  -/
  apply le_antisymm
    /-
      case a
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) c
      ha : LT.lt a c
      hc : LE.le Cardinal.aleph0 c
      ⊢ LE.le b c
    -/
  · rw [← h]
    /-
      case a
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) c
      ha : LT.lt a c
      hc : LE.le Cardinal.aleph0 c
      ⊢ LE.le b (HAdd.hAdd a b)
    -/
    apply self_le_add_left
    /-
      🎉 no goals
    -/
  /-
    case a
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd a b) c
    ha : LT.lt a c
    hc : LE.le Cardinal.aleph0 c
    ⊢ LE.le c b
  -/
  rw [← not_lt]; intro hb
  /-
    case a
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd a b) c
    ha : LT.lt a c
    hc : LE.le Cardinal.aleph0 c
    hb : LT.lt b c
    ⊢ False
  -/
  have : a + b < c := add_lt_of_lt hc ha hb
  /-
    case a
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd a b) c
    ha : LT.lt a c
    hc : LE.le Cardinal.aleph0 c
    hb : LT.lt b c
    this : LT.lt (HAdd.hAdd a b) c
    ⊢ False
  -/
  simp [h, lt_irrefl] at this
  /-
    🎉 no goals
  -/


theorem add_eq_left {a b : Cardinal} (ha : ℵ₀ ≤ a) (hb : b ≤ a) : a + b = a := by
  /-
    a b : Cardinal.{u_1}
    ha : LE.le Cardinal.aleph0 a
    hb : LE.le b a
    ⊢ Eq (HAdd.hAdd a b) a
  -/
  rw [add_eq_max ha, max_eq_left hb]
  /-
    🎉 no goals
  -/


theorem add_eq_right {a b : Cardinal} (hb : ℵ₀ ≤ b) (ha : a ≤ b) : a + b = b := by
  /-
    a b : Cardinal.{u_1}
    hb : LE.le Cardinal.aleph0 b
    ha : LE.le a b
    ⊢ Eq (HAdd.hAdd a b) b
  -/
  rw [add_comm, add_eq_left hb ha]
  /-
    🎉 no goals
  -/


theorem add_eq_left_iff {a b : Cardinal} : a + b = a ↔ max ℵ₀ b ≤ a ∨ b = 0 := by
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (Eq (HAdd.hAdd a b) a) (Or (LE.le (Max.max Cardinal.aleph0 b) a) (Eq b 0))
  -/
  rw [max_le_iff]
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (Eq (HAdd.hAdd a b) a) (Or (And (LE.le Cardinal.aleph0 a) (LE.le b a)) ( …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      a b : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) a
      ⊢ Or (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Eq b 0)
    -/
  · rcases le_or_lt ℵ₀ a with ha | ha
      /-
        case refine_1.inl
        a b : Cardinal.{u_1}
        h : Eq (HAdd.hAdd a b) a
        ha : LE.le Cardinal.aleph0 a
        ⊢ Or (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Eq b 0)
      -/
    · left
      /-
        case refine_1.inl.h
        a b : Cardinal.{u_1}
        h : Eq (HAdd.hAdd a b) a
        ha : LE.le Cardinal.aleph0 a
        ⊢ And (LE.le Cardinal.aleph0 a) (LE.le b a)
      -/
      use ha
      /-
        case right
        a b : Cardinal.{u_1}
        h : Eq (HAdd.hAdd a b) a
        ha : LE.le Cardinal.aleph0 a
        ⊢ LE.le b a
      -/
      rw [← not_lt]
      /-
        case right
        a b : Cardinal.{u_1}
        h : Eq (HAdd.hAdd a b) a
        ha : LE.le Cardinal.aleph0 a
        ⊢ Not (LT.lt a b)
      -/
      apply fun hb => ne_of_gt _ h
      /-
        a b : Cardinal.{u_1}
        h : Eq (HAdd.hAdd a b) a
        ha : LE.le Cardinal.aleph0 a
        ⊢ LT.lt a b → LT.lt a (HAdd.hAdd a b)
      -/
      intro hb
      /-
        a b : Cardinal.{u_1}
        h : Eq (HAdd.hAdd a b) a
        ha : LE.le Cardinal.aleph0 a
        hb : LT.lt a b
        ⊢ LT.lt a (HAdd.hAdd a b)
      -/
      exact hb.trans_le (self_le_add_left b a)
      /-
        🎉 no goals
      -/
    /-
      case refine_1.inr
      a b : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) a
      ha : LT.lt a Cardinal.aleph0
      ⊢ Or (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Eq b 0)
    -/
    right
    /-
      case refine_1.inr.h
      a b : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) a
      ha : LT.lt a Cardinal.aleph0
      ⊢ Eq b 0
    -/
    rw [← h, add_lt_aleph0_iff, lt_aleph0, lt_aleph0] at ha
    /-
      case refine_1.inr.h
      a b : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) a
      ha : And (Exists fun n => Eq a ↑n) (Exists fun n => Eq b ↑n)
      ⊢ Eq b 0
    -/
    rcases ha with ⟨⟨n, rfl⟩, ⟨m, rfl⟩⟩
    /-
      case refine_1.inr.h.intro.intro.intro
      n m : Nat
      h : Eq (HAdd.hAdd ↑n ↑m) ↑n
      ⊢ Eq (↑m) 0
    -/
    norm_cast at h ⊢
    /-
      case refine_1.inr.h.intro.intro.intro
      n m : Nat
      h : Eq (HAdd.hAdd n m) n
      ⊢ Eq m 0
    -/
    rw [← add_right_inj, h, add_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Cardinal.{u_1}
      ⊢ Or (And (LE.le Cardinal.aleph0 a) (LE.le b a)) (Eq b 0) → Eq (HAdd.hAdd a b) a
    -/
  · rintro (⟨h1, h2⟩ | h3)
      /-
        case refine_2.inl.intro
        a b : Cardinal.{u_1}
        h1 : LE.le Cardinal.aleph0 a
        h2 : LE.le b a
        ⊢ Eq (HAdd.hAdd a b) a
      -/
    · rw [add_eq_max h1, max_eq_left h2]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.inr
        a b : Cardinal.{u_1}
        h3 : Eq b 0
        ⊢ Eq (HAdd.hAdd a b) a
      -/
    · rw [h3, add_zero]
      /-
        🎉 no goals
      -/


theorem add_eq_right_iff {a b : Cardinal} : a + b = b ↔ max ℵ₀ a ≤ b ∨ a = 0 := by
  /-
    a b : Cardinal.{u_1}
    ⊢ Iff (Eq (HAdd.hAdd a b) b) (Or (LE.le (Max.max Cardinal.aleph0 a) b) (Eq a 0))
  -/
  rw [add_comm, add_eq_left_iff]
  /-
    🎉 no goals
  -/


theorem add_nat_eq {a : Cardinal} (n : ℕ) (ha : ℵ₀ ≤ a) : a + n = a :=
  add_eq_left ha ((nat_lt_aleph0 _).le.trans ha)


theorem nat_add_eq {a : Cardinal} (n : ℕ) (ha : ℵ₀ ≤ a) : n + a = a := by
  /-
    a : Cardinal.{u_1}
    n : Nat
    ha : LE.le Cardinal.aleph0 a
    ⊢ Eq (HAdd.hAdd (↑n) a) a
  -/
  rw [add_comm, add_nat_eq n ha]
  /-
    🎉 no goals
  -/


theorem add_one_eq {a : Cardinal} (ha : ℵ₀ ≤ a) : a + 1 = a :=
  add_one_of_aleph0_le ha


theorem mk_add_one_eq {α : Type*} [Infinite α] : #α + 1 = #α :=
  add_one_eq (aleph0_le_mk α)


protected theorem eq_of_add_eq_add_left {a b c : Cardinal} (h : a + b = a + c) (ha : a < ℵ₀) :
    b = c := by
  /-
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
    ha : LT.lt a Cardinal.aleph0
    ⊢ Eq b c
  -/
  rcases le_or_lt ℵ₀ b with hb | hb
    /-
      case inl
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
      ha : LT.lt a Cardinal.aleph0
      hb : LE.le Cardinal.aleph0 b
      ⊢ Eq b c
    -/
  · have : a < b := ha.trans_le hb
    /-
      case inl
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
      ha : LT.lt a Cardinal.aleph0
      hb : LE.le Cardinal.aleph0 b
      this : LT.lt a b
      ⊢ Eq b c
    -/
    rw [add_eq_right hb this.le, eq_comm] at h
    /-
      case inl
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a c) b
      ha : LT.lt a Cardinal.aleph0
      hb : LE.le Cardinal.aleph0 b
      this : LT.lt a b
      ⊢ Eq b c
    -/
    rw [eq_of_add_eq_of_aleph0_le h this hb]
    /-
      🎉 no goals
    -/
  · have hc : c < ℵ₀ := by
      rw [← not_le]
      intro hc
      apply lt_irrefl ℵ₀
      apply (hc.trans (self_le_add_left _ a)).trans_lt
      rw [← h]
      apply add_lt_aleph0 ha hb
    /-
      case inr
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
      ha : LT.lt a Cardinal.aleph0
      hb : LT.lt b Cardinal.aleph0
      hc : LT.lt c Cardinal.aleph0
      ⊢ Eq b c
    -/
    rw [lt_aleph0] at *
    /-
      case inr
      a b c : Cardinal.{u_1}
      h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
      ha : Exists fun n => Eq a ↑n
      hb : Exists fun n => Eq b ↑n
      hc : Exists fun n => Eq c ↑n
      ⊢ Eq b c
    -/
    rcases ha with ⟨n, rfl⟩
    /-
      case inr.intro
      b c : Cardinal.{u_1}
      hb : Exists fun n => Eq b ↑n
      hc : Exists fun n => Eq c ↑n
      n : Nat
      h : Eq (HAdd.hAdd (↑n) b) (HAdd.hAdd (↑n) c)
      ⊢ Eq b c
    -/
    rcases hb with ⟨m, rfl⟩
    /-
      case inr.intro.intro
      c : Cardinal.{u_1}
      hc : Exists fun n => Eq c ↑n
      n m : Nat
      h : Eq (HAdd.hAdd ↑n ↑m) (HAdd.hAdd (↑n) c)
      ⊢ Eq (↑m) c
    -/
    rcases hc with ⟨k, rfl⟩
    /-
      case inr.intro.intro.intro
      n m k : Nat
      h : Eq (HAdd.hAdd ↑n ↑m) (HAdd.hAdd ↑n ↑k)
      ⊢ Eq ↑m ↑k
    -/
    norm_cast at h ⊢
    /-
      case inr.intro.intro.intro
      n m k : Nat
      h : Eq (HAdd.hAdd n m) (HAdd.hAdd n k)
      ⊢ Eq m k
    -/
    apply add_left_cancel h
    /-
      🎉 no goals
    -/


protected theorem eq_of_add_eq_add_right {a b c : Cardinal} (h : a + b = c + b) (hb : b < ℵ₀) :
    a = c := by
  /-
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd a b) (HAdd.hAdd c b)
    hb : LT.lt b Cardinal.aleph0
    ⊢ Eq a c
  -/
  rw [add_comm a b, add_comm c b] at h
  /-
    a b c : Cardinal.{u_1}
    h : Eq (HAdd.hAdd b a) (HAdd.hAdd b c)
    hb : LT.lt b Cardinal.aleph0
    ⊢ Eq a c
  -/
  exact Cardinal.eq_of_add_eq_add_left h hb
  /-
    🎉 no goals
  -/


protected theorem ciSup_add (hf : BddAbove (range f)) (c : Cardinal.{v}) :
    (⨆ i, f i) + c = ⨆ i, f i + c := by
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    inst✝ : Nonempty ι
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    ⊢ Eq (HAdd.hAdd (iSup fun i => f i) c) (iSup fun i => HAdd.hAdd (f i) c)
  -/
  have : ∀ i, f i + c ≤ (⨆ i, f i) + c := fun i ↦ add_le_add_right (le_ciSup hf i) c
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    inst✝ : Nonempty ι
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    this : ∀ (i : ι), LE.le (HAdd.hAdd (f i) c) (HAdd.hAdd (iSup fun i => f i) c)
    ⊢ Eq (HAdd.hAdd (iSup fun i => f i) c) (iSup fun i => HAdd.hAdd (f i) c)
  -/
  refine le_antisymm ?_ (ciSup_le' this)
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    inst✝ : Nonempty ι
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    this : ∀ (i : ι), LE.le (HAdd.hAdd (f i) c) (HAdd.hAdd (iSup fun i => f i) c)
    ⊢ LE.le (HAdd.hAdd (iSup fun i => f i) c) (iSup fun i => HAdd.hAdd (f i) c)
  -/
  have bdd : BddAbove (range (f · + c)) := ⟨_, forall_mem_range.mpr this⟩
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    inst✝ : Nonempty ι
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    this : ∀ (i : ι), LE.le (HAdd.hAdd (f i) c) (HAdd.hAdd (iSup fun i => f i) c)
    bdd : BddAbove (Set.range fun x => HAdd.hAdd (f x) c)
    ⊢ LE.le (HAdd.hAdd (iSup fun i => f i) c) (iSup fun i => HAdd.hAdd (f i) c)
  -/
  obtain hs | hs := lt_or_le (⨆ i, f i) ℵ₀
  · obtain ⟨i, hi⟩ := exists_eq_of_iSup_eq_of_not_isSuccLimit
      f hf (not_isSuccLimit_of_lt_aleph0 hs) rfl
    /-
      case inl.intro
      ι : Type u
      f : ι → Cardinal.{v}
      inst✝ : Nonempty ι
      hf : BddAbove (Set.range f)
      c : Cardinal.{v}
      this : ∀ (i : ι), LE.le (HAdd.hAdd (f i) c) (HAdd.hAdd (iSup fun i => f i) c)
      bdd : BddAbove (Set.range fun x => HAdd.hAdd (f x) c)
      hs : LT.lt (iSup fun i => f i) Cardinal.aleph0
      i : ι
      hi : Eq (f i) (iSup fun i => f i)
      ⊢ LE.le (HAdd.hAdd (iSup fun i => f i) c) (iSup fun i => HAdd.hAdd (f i) c)
    -/
    exact hi ▸ le_ciSup bdd i
    /-
      🎉 no goals
    -/
  /-
    case inr
    ι : Type u
    f : ι → Cardinal.{v}
    inst✝ : Nonempty ι
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    this : ∀ (i : ι), LE.le (HAdd.hAdd (f i) c) (HAdd.hAdd (iSup fun i => f i) c)
    bdd : BddAbove (Set.range fun x => HAdd.hAdd (f x) c)
    hs : LE.le Cardinal.aleph0 (iSup fun i => f i)
    ⊢ LE.le (HAdd.hAdd (iSup fun i => f i) c) (iSup fun i => HAdd.hAdd (f i) c)
  -/
  rw [add_eq_max hs, max_le_iff]
  exact ⟨ciSup_mono bdd fun i ↦ self_le_add_right _ c,
    (self_le_add_left _ _).trans (le_ciSup bdd <| Classical.arbitrary ι)⟩


protected theorem add_ciSup (hf : BddAbove (range f)) (c : Cardinal.{v}) :
    c + (⨆ i, f i) = ⨆ i, c + f i := by
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    inst✝ : Nonempty ι
    hf : BddAbove (Set.range f)
    c : Cardinal.{v}
    ⊢ Eq (HAdd.hAdd c (iSup fun i => f i)) (iSup fun i => HAdd.hAdd c (f i))
  -/
  rw [add_comm, Cardinal.ciSup_add f hf]; simp_rw [add_comm]
                                          /-
                                            🎉 no goals
                                          -/


protected theorem ciSup_add_ciSup (hf : BddAbove (range f)) (g : ι' → Cardinal.{v})
    (hg : BddAbove (range g)) :
    (⨆ i, f i) + (⨆ j, g j) = ⨆ (i) (j), f i + g j := by
  /-
    ι : Type u
    ι' : Type w
    f : ι → Cardinal.{v}
    inst✝¹ : Nonempty ι
    inst✝ : Nonempty ι'
    hf : BddAbove (Set.range f)
    g : ι' → Cardinal.{v}
    hg : BddAbove (Set.range g)
    ⊢ Eq (HAdd.hAdd (iSup fun i => f i) (iSup fun j => g j)) (iSup fun i => iSup f …
  -/
  simp_rw [Cardinal.ciSup_add f hf, Cardinal.add_ciSup g hg]
  /-
    🎉 no goals
  -/


protected theorem ciSup_mul (c : Cardinal.{v}) : (⨆ i, f i) * c = ⨆ i, f i * c := by
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    ⊢ Eq (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  cases isEmpty_or_nonempty ι; · simp
                                 /-
                                   🎉 no goals
                                 -/
  /-
    case inr
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    ⊢ Eq (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  obtain rfl | h0 := eq_or_ne c 0; · simp
                                     /-
                                       🎉 no goals
                                     -/
  /-
    case inr.inr
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    ⊢ Eq (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  by_cases hf : BddAbove (range f); swap
  · have hfc : ¬ BddAbove (range (f · * c)) := fun bdd ↦ hf
      ⟨⨆ i, f i * c, forall_mem_range.mpr fun i ↦ (le_mul_right h0).trans (le_ciSup bdd i)⟩
    /-
      case neg
      ι : Type u
      f : ι → Cardinal.{v}
      c : Cardinal.{v}
      h✝ : Nonempty ι
      h0 : Ne c 0
      hf : Not (BddAbove (Set.range f))
      hfc : Not (BddAbove (Set.range fun x => HMul.hMul (f x) c))
      ⊢ Eq (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
    -/
    simp [iSup, csSup_of_not_bddAbove, hf, hfc]
    /-
      🎉 no goals
    -/
  /-
    case pos
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    hf : BddAbove (Set.range f)
    ⊢ Eq (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  have : ∀ i, f i * c ≤ (⨆ i, f i) * c := fun i ↦ mul_le_mul_right' (le_ciSup hf i) c
  /-
    case pos
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    hf : BddAbove (Set.range f)
    this : ∀ (i : ι), LE.le (HMul.hMul (f i) c) (HMul.hMul (iSup fun i => f i) c)
    ⊢ Eq (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  refine le_antisymm ?_ (ciSup_le' this)
  /-
    case pos
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    hf : BddAbove (Set.range f)
    this : ∀ (i : ι), LE.le (HMul.hMul (f i) c) (HMul.hMul (iSup fun i => f i) c)
    ⊢ LE.le (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  have bdd : BddAbove (range (f · * c)) := ⟨_, forall_mem_range.mpr this⟩
  /-
    case pos
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    hf : BddAbove (Set.range f)
    this : ∀ (i : ι), LE.le (HMul.hMul (f i) c) (HMul.hMul (iSup fun i => f i) c)
    bdd : BddAbove (Set.range fun x => HMul.hMul (f x) c)
    ⊢ LE.le (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  obtain hs | hs := lt_or_le (⨆ i, f i) ℵ₀
  · obtain ⟨i, hi⟩ := exists_eq_of_iSup_eq_of_not_isSuccLimit
      f hf (not_isSuccLimit_of_lt_aleph0 hs) rfl
    /-
      case pos.inl.intro
      ι : Type u
      f : ι → Cardinal.{v}
      c : Cardinal.{v}
      h✝ : Nonempty ι
      h0 : Ne c 0
      hf : BddAbove (Set.range f)
      this : ∀ (i : ι), LE.le (HMul.hMul (f i) c) (HMul.hMul (iSup fun i => f i) c)
      bdd : BddAbove (Set.range fun x => HMul.hMul (f x) c)
      hs : LT.lt (iSup fun i => f i) Cardinal.aleph0
      i : ι
      hi : Eq (f i) (iSup fun i => f i)
      ⊢ LE.le (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
    -/
    exact hi ▸ le_ciSup bdd i
    /-
      🎉 no goals
    -/
  /-
    case pos.inr
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    hf : BddAbove (Set.range f)
    this : ∀ (i : ι), LE.le (HMul.hMul (f i) c) (HMul.hMul (iSup fun i => f i) c)
    bdd : BddAbove (Set.range fun x => HMul.hMul (f x) c)
    hs : LE.le Cardinal.aleph0 (iSup fun i => f i)
    ⊢ LE.le (HMul.hMul (iSup fun i => f i) c) (iSup fun i => HMul.hMul (f i) c)
  -/
  rw [mul_eq_max_of_aleph0_le_left hs h0, max_le_iff]
  /-
    case pos.inr
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    h✝ : Nonempty ι
    h0 : Ne c 0
    hf : BddAbove (Set.range f)
    this : ∀ (i : ι), LE.le (HMul.hMul (f i) c) (HMul.hMul (iSup fun i => f i) c)
    bdd : BddAbove (Set.range fun x => HMul.hMul (f x) c)
    hs : LE.le Cardinal.aleph0 (iSup fun i => f i)
    ⊢ And (LE.le (iSup fun i => f i) (iSup fun i => HMul.hMul (f i) c)) (LE.le c ( …
  -/
  obtain ⟨i, hi⟩ := exists_lt_of_lt_ciSup' (one_lt_aleph0.trans_le hs)
  exact ⟨ciSup_mono bdd fun i ↦ le_mul_right h0,
    (le_mul_left (zero_lt_one.trans hi).ne').trans (le_ciSup bdd i)⟩


protected theorem mul_ciSup (c : Cardinal.{v}) : c * (⨆ i, f i) = ⨆ i, c * f i := by
  /-
    ι : Type u
    f : ι → Cardinal.{v}
    c : Cardinal.{v}
    ⊢ Eq (HMul.hMul c (iSup fun i => f i)) (iSup fun i => HMul.hMul c (f i))
  -/
  rw [mul_comm, Cardinal.ciSup_mul f]; simp_rw [mul_comm]
                                       /-
                                         🎉 no goals
                                       -/


protected theorem ciSup_mul_ciSup (g : ι' → Cardinal.{v}) :
    (⨆ i, f i) * (⨆ j, g j) = ⨆ (i) (j), f i * g j := by
  /-
    ι : Type u
    ι' : Type w
    f : ι → Cardinal.{v}
    g : ι' → Cardinal.{v}
    ⊢ Eq (HMul.hMul (iSup fun i => f i) (iSup fun j => g j)) (iSup fun i => iSup f …
  -/
  simp_rw [Cardinal.ciSup_mul f, Cardinal.mul_ciSup g]
  /-
    🎉 no goals
  -/


theorem sum_eq_iSup_lift {f : ι → Cardinal.{max u v}} (hι : ℵ₀ ≤ #ι)
    (h : lift.{v} #ι ≤ iSup f) : sum f = iSup f := by
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    hι : LE.le Cardinal.aleph0 (Cardinal.mk ι)
    h : LE.le (Cardinal.lift.{v, u} (Cardinal.mk ι)) (iSup f)
    ⊢ Eq (Cardinal.sum f) (iSup f)
  -/
  apply (iSup_le_sum f).antisymm'
  /-
    ι : Type u
    f : ι → Cardinal.{max u v}
    hι : LE.le Cardinal.aleph0 (Cardinal.mk ι)
    h : LE.le (Cardinal.lift.{v, u} (Cardinal.mk ι)) (iSup f)
    ⊢ LE.le (Cardinal.sum f) (iSup f)
  -/
  convert sum_le_iSup_lift f
  /-
    case h.e'_4
    ι : Type u
    f : ι → Cardinal.{max u v}
    hι : LE.le Cardinal.aleph0 (Cardinal.mk ι)
    h : LE.le (Cardinal.lift.{v, u} (Cardinal.mk ι)) (iSup f)
    ⊢ Eq (iSup f) (HMul.hMul (Cardinal.lift.{v, u} (Cardinal.mk ι)) (iSup f))
  -/
  rw [mul_eq_max (aleph0_le_lift.mpr hι) ((aleph0_le_lift.mpr hι).trans h), max_eq_right h]
  /-
    🎉 no goals
  -/


theorem sum_eq_iSup {f : ι → Cardinal.{u}} (hι : ℵ₀ ≤ #ι) (h : #ι ≤ iSup f) : sum f = iSup f :=
  sum_eq_iSup_lift hι ((lift_id #ι).symm ▸ h)


@[simp]
theorem aleph_add_aleph (o₁ o₂ : Ordinal) : ℵ_ o₁ + ℵ_ o₂ = ℵ_ (max o₁ o₂) := by
  /-
    o₁ o₂ : Ordinal.{u_1}
    ⊢ Eq (HAdd.hAdd (Cardinal.aleph o₁) (Cardinal.aleph o₂)) (Cardinal.aleph (Max. …
  -/
  rw [Cardinal.add_eq_max (aleph0_le_aleph o₁), aleph_max]
  /-
    🎉 no goals
  -/


theorem principal_add_ord {c : Cardinal} (hc : ℵ₀ ≤ c) : Ordinal.Principal (· + ·) c.ord :=
  fun a b ha hb => by
  /-
    c : Cardinal.{u_1}
    hc : LE.le Cardinal.aleph0 c
    a b : Ordinal.{u_1}
    ha : LT.lt a c.ord
    hb : LT.lt b c.ord
    ⊢ LT.lt ((fun x1 x2 => HAdd.hAdd x1 x2) a b) c.ord
  -/
  rw [lt_ord, Ordinal.card_add] at *
  /-
    c : Cardinal.{u_1}
    hc : LE.le Cardinal.aleph0 c
    a b : Ordinal.{u_1}
    ha : LT.lt a.card c
    hb : LT.lt b.card c
    ⊢ LT.lt (HAdd.hAdd a.card b.card) c
  -/
  exact add_lt_of_lt hc ha hb
  /-
    🎉 no goals
  -/


theorem principal_add_aleph (o : Ordinal) : Ordinal.Principal (· + ·) (ℵ_ o).ord :=
  principal_add_ord <| aleph0_le_aleph o


theorem add_right_inj_of_lt_aleph0 {α β γ : Cardinal} (γ₀ : γ < aleph0) : α + γ = β + γ ↔ α = β :=
  ⟨fun h => Cardinal.eq_of_add_eq_add_right h γ₀, fun h => congr_arg (· + γ) h⟩


@[simp]
theorem add_nat_inj {α β : Cardinal} (n : ℕ) : α + n = β + n ↔ α = β :=
  add_right_inj_of_lt_aleph0 (nat_lt_aleph0 _)


@[simp]
theorem add_one_inj {α β : Cardinal} : α + 1 = β + 1 ↔ α = β :=
  add_right_inj_of_lt_aleph0 one_lt_aleph0


theorem add_le_add_iff_of_lt_aleph0 {α β γ : Cardinal} (γ₀ : γ < ℵ₀) :
    α + γ ≤ β + γ ↔ α ≤ β := by
  /-
    α β γ : Cardinal.{u_1}
    γ₀ : LT.lt γ Cardinal.aleph0
    ⊢ Iff (LE.le (HAdd.hAdd α γ) (HAdd.hAdd β γ)) (LE.le α β)
  -/
  refine ⟨fun h => ?_, fun h => add_le_add_right h γ⟩
  /-
    α β γ : Cardinal.{u_1}
    γ₀ : LT.lt γ Cardinal.aleph0
    h : LE.le (HAdd.hAdd α γ) (HAdd.hAdd β γ)
    ⊢ LE.le α β
  -/
  contrapose h
  /-
    α β γ : Cardinal.{u_1}
    γ₀ : LT.lt γ Cardinal.aleph0
    h : Not (LE.le α β)
    ⊢ Not (LE.le (HAdd.hAdd α γ) (HAdd.hAdd β γ))
  -/
  rw [not_le, lt_iff_le_and_ne, Ne] at h ⊢
  /-
    α β γ : Cardinal.{u_1}
    γ₀ : LT.lt γ Cardinal.aleph0
    h : And (LE.le β α) (Not (Eq β α))
    ⊢ And (LE.le (HAdd.hAdd β γ) (HAdd.hAdd α γ)) (Not (Eq (HAdd.hAdd β γ) (HAdd.h …
  -/
  exact ⟨add_le_add_right h.1 γ, mt (add_right_inj_of_lt_aleph0 γ₀).1 h.2⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem add_nat_le_add_nat_iff {α β : Cardinal} (n : ℕ) : α + n ≤ β + n ↔ α ≤ β :=
  add_le_add_iff_of_lt_aleph0 (nat_lt_aleph0 n)


@[deprecated (since := "2024-02-12")]
alias add_nat_le_add_nat_iff_of_lt_aleph_0 := add_nat_le_add_nat_iff


@[simp]
theorem add_one_le_add_one_iff {α β : Cardinal} : α + 1 ≤ β + 1 ↔ α ≤ β :=
  add_le_add_iff_of_lt_aleph0 one_lt_aleph0


@[deprecated (since := "2024-02-12")]
alias add_one_le_add_one_iff_of_lt_aleph_0 := add_one_le_add_one_iff


theorem pow_le {κ μ : Cardinal.{u}} (H1 : ℵ₀ ≤ κ) (H2 : μ < ℵ₀) : κ ^ μ ≤ κ :=
  let ⟨n, H3⟩ := lt_aleph0.1 H2
  H3.symm ▸
    Quotient.inductionOn κ
      (fun α H1 =>
        Nat.recOn n
          (lt_of_lt_of_le
              (by
                /-
                  κ μ : Cardinal.{u}
                  H1✝ : LE.le Cardinal.aleph0 κ
                  H2 : LT.lt μ Cardinal.aleph0
                  n : Nat
                  H3 : Eq μ ↑n
                  α : Type u
                  H1 : LE.le Cardinal.aleph0 (Quotient.mk Cardinal.isEquivalent α)
                  ⊢ LT.lt (HPow.hPow (Quotient.mk Cardinal.isEquivalent α) ↑Nat.zero) Cardinal.a …
                -/
                rw [Nat.cast_zero, power_zero]
                /-
                  κ μ : Cardinal.{u}
                  H1✝ : LE.le Cardinal.aleph0 κ
                  H2 : LT.lt μ Cardinal.aleph0
                  n : Nat
                  H3 : Eq μ ↑n
                  α : Type u
                  H1 : LE.le Cardinal.aleph0 (Quotient.mk Cardinal.isEquivalent α)
                  ⊢ LT.lt 1 Cardinal.aleph0
                -/
                exact one_lt_aleph0)
                /-
                  🎉 no goals
                -/
              H1).le
          fun n ih =>
          le_of_le_of_eq
            (by
              /-
                κ μ : Cardinal.{u}
                H1✝ : LE.le Cardinal.aleph0 κ
                H2 : LT.lt μ Cardinal.aleph0
                n✝ : Nat
                H3 : Eq μ ↑n✝
                α : Type u
                H1 : LE.le Cardinal.aleph0 (Quotient.mk Cardinal.isEquivalent α)
                n : Nat
                ih : LE.le (HPow.hPow (Quotient.mk Cardinal.isEquivalent α) ↑n) (Quotient.mk C …
                ⊢ LE.le (HPow.hPow (Quotient.mk Cardinal.isEquivalent α) ↑n.succ) (HMul.hMul ( …
              -/
              rw [Nat.cast_succ, power_add, power_one]
              /-
                κ μ : Cardinal.{u}
                H1✝ : LE.le Cardinal.aleph0 κ
                H2 : LT.lt μ Cardinal.aleph0
                n✝ : Nat
                H3 : Eq μ ↑n✝
                α : Type u
                H1 : LE.le Cardinal.aleph0 (Quotient.mk Cardinal.isEquivalent α)
                n : Nat
                ih : LE.le (HPow.hPow (Quotient.mk Cardinal.isEquivalent α) ↑n) (Quotient.mk C …
                ⊢ LE.le (HMul.hMul (HPow.hPow (Quotient.mk Cardinal.isEquivalent α) ↑n) (Quoti …
              -/
              exact mul_le_mul_right' ih _)
              /-
                🎉 no goals
              -/
            (mul_eq_self H1))
      H1


theorem pow_eq {κ μ : Cardinal.{u}} (H1 : ℵ₀ ≤ κ) (H2 : 1 ≤ μ) (H3 : μ < ℵ₀) : κ ^ μ = κ :=
  (pow_le H1 H3).antisymm <| self_le_power κ H2


theorem power_self_eq {c : Cardinal} (h : ℵ₀ ≤ c) : c ^ c = 2 ^ c := by
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ Eq (HPow.hPow c c) (HPow.hPow 2 c)
  -/
  apply ((power_le_power_right <| (cantor c).le).trans _).antisymm
    /-
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      ⊢ LE.le (HPow.hPow 2 c) (HPow.hPow c c)
    -/
  · exact power_le_power_right ((nat_lt_aleph0 2).le.trans h)
    /-
      🎉 no goals
    -/
    /-
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      ⊢ LE.le (HPow.hPow (HPow.hPow 2 c) c) (HPow.hPow 2 c)
    -/
  · rw [← power_mul, mul_eq_self h]
    /-
      🎉 no goals
    -/


theorem prod_eq_two_power {ι : Type u} [Infinite ι] {c : ι → Cardinal.{v}} (h₁ : ∀ i, 2 ≤ c i)
    (h₂ : ∀ i, lift.{u} (c i) ≤ lift.{v} #ι) : prod c = 2 ^ lift.{v} #ι := by
  /-
    ι : Type u
    inst✝ : Infinite ι
    c : ι → Cardinal.{v}
    h₁ : ∀ (i : ι), LE.le 2 (c i)
    h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
    ⊢ Eq (Cardinal.prod c) (HPow.hPow 2 (Cardinal.lift.{v, u} (Cardinal.mk ι)))
  -/
  rw [← lift_id'.{u, v} (prod.{u, v} c), lift_prod, ← lift_two_power]
  /-
    ι : Type u
    inst✝ : Infinite ι
    c : ι → Cardinal.{v}
    h₁ : ∀ (i : ι), LE.le 2 (c i)
    h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
    ⊢ Eq (Cardinal.prod fun i => Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} …
  -/
  apply le_antisymm
    /-
      case a
      ι : Type u
      inst✝ : Infinite ι
      c : ι → Cardinal.{v}
      h₁ : ∀ (i : ι), LE.le 2 (c i)
      h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
      ⊢ LE.le (Cardinal.prod fun i => Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, …
    -/
  · refine (prod_le_prod _ _ h₂).trans_eq ?_
    /-
      case a
      ι : Type u
      inst✝ : Infinite ι
      c : ι → Cardinal.{v}
      h₁ : ∀ (i : ι), LE.le 2 (c i)
      h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
      ⊢ Eq (Cardinal.prod fun i => Cardinal.lift.{v, u} (Cardinal.mk ι)) (Cardinal.l …
    -/
    rw [prod_const, lift_lift, ← lift_power, power_self_eq (aleph0_le_mk ι), lift_umax.{u, v}]
    /-
      🎉 no goals
    -/
    /-
      case a
      ι : Type u
      inst✝ : Infinite ι
      c : ι → Cardinal.{v}
      h₁ : ∀ (i : ι), LE.le 2 (c i)
      h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
      ⊢ LE.le (Cardinal.lift.{v, u} (HPow.hPow 2 (Cardinal.mk ι))) (Cardinal.prod fu …
    -/
  · rw [← prod_const', lift_prod]
    /-
      case a
      ι : Type u
      inst✝ : Infinite ι
      c : ι → Cardinal.{v}
      h₁ : ∀ (i : ι), LE.le 2 (c i)
      h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
      ⊢ LE.le (Cardinal.prod fun i => Cardinal.lift.{v, u} 2) (Cardinal.prod fun i = …
    -/
    refine prod_le_prod _ _ fun i => ?_
    /-
      case a
      ι : Type u
      inst✝ : Infinite ι
      c : ι → Cardinal.{v}
      h₁ : ∀ (i : ι), LE.le 2 (c i)
      h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
      i : ι
      ⊢ LE.le (Cardinal.lift.{v, u} 2) (Cardinal.lift.{u, v} (c i))
    -/
    rw [lift_two, ← lift_two.{u, v}, lift_le]
    /-
      case a
      ι : Type u
      inst✝ : Infinite ι
      c : ι → Cardinal.{v}
      h₁ : ∀ (i : ι), LE.le 2 (c i)
      h₂ : ∀ (i : ι), LE.le (Cardinal.lift.{u, v} (c i)) (Cardinal.lift.{v, u} (Card …
      i : ι
      ⊢ LE.le 2 (c i)
    -/
    exact h₁ i
    /-
      🎉 no goals
    -/


theorem power_eq_two_power {c₁ c₂ : Cardinal} (h₁ : ℵ₀ ≤ c₁) (h₂ : 2 ≤ c₂) (h₂' : c₂ ≤ c₁) :
    c₂ ^ c₁ = 2 ^ c₁ :=
  le_antisymm (power_self_eq h₁ ▸ power_le_power_right h₂') (power_le_power_right h₂)


theorem nat_power_eq {c : Cardinal.{u}} (h : ℵ₀ ≤ c) {n : ℕ} (hn : 2 ≤ n) :
    (n : Cardinal.{u}) ^ c = 2 ^ c :=
                           /-
                             c : Cardinal.{u}
                             h : LE.le Cardinal.aleph0 c
                             n : Nat
                             hn : LE.le 2 n
                             ⊢ LE.le 2 ↑n
                           -/
  power_eq_two_power h (by assumption_mod_cast) ((nat_lt_aleph0 n).le.trans h)
                           /-
                             🎉 no goals
                           -/


theorem power_nat_le {c : Cardinal.{u}} {n : ℕ} (h : ℵ₀ ≤ c) : c ^ n ≤ c :=
  pow_le h (nat_lt_aleph0 n)


theorem power_nat_eq {c : Cardinal.{u}} {n : ℕ} (h1 : ℵ₀ ≤ c) (h2 : 1 ≤ n) : c ^ n = c :=
  pow_eq h1 (mod_cast h2) (nat_lt_aleph0 n)


theorem power_nat_le_max {c : Cardinal.{u}} {n : ℕ} : c ^ (n : Cardinal.{u}) ≤ max c ℵ₀ := by
  /-
    c : Cardinal.{u}
    n : Nat
    ⊢ LE.le (HPow.hPow c ↑n) (Max.max c Cardinal.aleph0)
  -/
  rcases le_or_lt ℵ₀ c with hc | hc
    /-
      case inl
      c : Cardinal.{u}
      n : Nat
      hc : LE.le Cardinal.aleph0 c
      ⊢ LE.le (HPow.hPow c ↑n) (Max.max c Cardinal.aleph0)
    -/
  · exact le_max_of_le_left (power_nat_le hc)
    /-
      🎉 no goals
    -/
    /-
      case inr
      c : Cardinal.{u}
      n : Nat
      hc : LT.lt c Cardinal.aleph0
      ⊢ LE.le (HPow.hPow c ↑n) (Max.max c Cardinal.aleph0)
    -/
  · exact le_max_of_le_right (power_lt_aleph0 hc (nat_lt_aleph0 _)).le
    /-
      🎉 no goals
    -/


theorem powerlt_aleph0 {c : Cardinal} (h : ℵ₀ ≤ c) : c ^< ℵ₀ = c := by
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ Eq (c.powerlt Cardinal.aleph0) c
  -/
  apply le_antisymm
    /-
      case a
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      ⊢ LE.le (c.powerlt Cardinal.aleph0) c
    -/
  · rw [powerlt_le]
    /-
      case a
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      ⊢ ∀ (x : Cardinal.{u_1}), LT.lt x Cardinal.aleph0 → LE.le (HPow.hPow c x) c
    -/
    intro c'
    /-
      case a
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      c' : Cardinal.{u_1}
      ⊢ LT.lt c' Cardinal.aleph0 → LE.le (HPow.hPow c c') c
    -/
    rw [lt_aleph0]
    /-
      case a
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      c' : Cardinal.{u_1}
      ⊢ (Exists fun n => Eq c' ↑n) → LE.le (HPow.hPow c c') c
    -/
    rintro ⟨n, rfl⟩
    /-
      case a.intro
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      n : Nat
      ⊢ LE.le (HPow.hPow c ↑n) c
    -/
    apply power_nat_le h
    /-
      🎉 no goals
    -/
  /-
    case a
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ LE.le c (c.powerlt Cardinal.aleph0)
  -/
  convert le_powerlt c one_lt_aleph0; rw [power_one]
                                      /-
                                        🎉 no goals
                                      -/


theorem powerlt_aleph0_le (c : Cardinal) : c ^< ℵ₀ ≤ max c ℵ₀ := by
  /-
    c : Cardinal.{u_1}
    ⊢ LE.le (c.powerlt Cardinal.aleph0) (Max.max c Cardinal.aleph0)
  -/
  rcases le_or_lt ℵ₀ c with h | h
    /-
      case inl
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      ⊢ LE.le (c.powerlt Cardinal.aleph0) (Max.max c Cardinal.aleph0)
    -/
  · rw [powerlt_aleph0 h]
    /-
      case inl
      c : Cardinal.{u_1}
      h : LE.le Cardinal.aleph0 c
      ⊢ LE.le c (Max.max c Cardinal.aleph0)
    -/
    apply le_max_left
    /-
      🎉 no goals
    -/
  /-
    case inr
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ LE.le (c.powerlt Cardinal.aleph0) (Max.max c Cardinal.aleph0)
  -/
  rw [powerlt_le]
  /-
    case inr
    c : Cardinal.{u_1}
    h : LT.lt c Cardinal.aleph0
    ⊢ ∀ (x : Cardinal.{u_1}), LT.lt x Cardinal.aleph0 → LE.le (HPow.hPow c x) (Max …
  -/
  exact fun c' hc' => (power_lt_aleph0 h hc').le.trans (le_max_right _ _)
  /-
    🎉 no goals
  -/


theorem mk_equiv_eq_zero_iff_lift_ne : #(α ≃ β') = 0 ↔ lift.{v} #α ≠ lift.{u} #β' := by
  /-
    α : Type u
    β' : Type v
    ⊢ Iff (Eq (Cardinal.mk (Equiv α β')) 0) (Ne (Cardinal.lift.{v, u} (Cardinal.mk …
  -/
  rw [mk_eq_zero_iff, ← not_nonempty_iff, ← lift_mk_eq']
  /-
    🎉 no goals
  -/


theorem mk_equiv_eq_zero_iff_ne : #(α ≃ β) = 0 ↔ #α ≠ #β := by
  /-
    α β : Type u
    ⊢ Iff (Eq (Cardinal.mk (Equiv α β)) 0) (Ne (Cardinal.mk α) (Cardinal.mk β))
  -/
  rw [mk_equiv_eq_zero_iff_lift_ne, lift_id, lift_id]
  /-
    🎉 no goals
  -/


/-- This lemma makes lemmas assuming `Infinite α` applicable to the situation where we have
  `Infinite β` instead. -/
theorem mk_equiv_comm : #(α ≃ β') = #(β' ≃ α) :=
  (ofBijective _ symm_bijective).cardinal_eq


theorem mk_embedding_eq_zero_iff_lift_lt : #(α ↪ β') = 0 ↔ lift.{u} #β' < lift.{v} #α := by
  /-
    α : Type u
    β' : Type v
    ⊢ Iff (Eq (Cardinal.mk (Function.Embedding α β')) 0) (LT.lt (Cardinal.lift.{u, …
  -/
  rw [mk_eq_zero_iff, ← not_nonempty_iff, ← lift_mk_le', not_le]
  /-
    🎉 no goals
  -/


theorem mk_embedding_eq_zero_iff_lt : #(α ↪ β) = 0 ↔ #β < #α := by
  /-
    α β : Type u
    ⊢ Iff (Eq (Cardinal.mk (Function.Embedding α β)) 0) (LT.lt (Cardinal.mk β) (Ca …
  -/
  rw [mk_embedding_eq_zero_iff_lift_lt, lift_lt]
  /-
    🎉 no goals
  -/


theorem mk_arrow_eq_zero_iff : #(α → β') = 0 ↔ #α ≠ 0 ∧ #β' = 0 := by
  /-
    α : Type u
    β' : Type v
    ⊢ Iff (Eq (Cardinal.mk (α → β')) 0) (And (Ne (Cardinal.mk α) 0) (Eq (Cardinal. …
  -/
  simp_rw [mk_eq_zero_iff, mk_ne_zero_iff, isEmpty_fun]
  /-
    🎉 no goals
  -/


theorem mk_surjective_eq_zero_iff_lift :
    #{f : α → β' | Surjective f} = 0 ↔ lift.{v} #α < lift.{u} #β' ∨ (#α ≠ 0 ∧ #β' = 0) := by
  /-
    α : Type u
    β' : Type v
    ⊢ Iff (Eq (Cardinal.mk ↑(setOf fun f => Function.Surjective f)) 0) (Or (LT.lt  …
  -/
  rw [← not_iff_not, not_or, not_lt, lift_mk_le', ← Ne, not_and_or, not_ne_iff, and_comm]
  simp_rw [mk_ne_zero_iff, mk_eq_zero_iff, nonempty_coe_sort,
    Set.Nonempty, mem_setOf, exists_surjective_iff, nonempty_fun]


theorem mk_surjective_eq_zero_iff :
    #{f : α → β | Surjective f} = 0 ↔ #α < #β ∨ (#α ≠ 0 ∧ #β = 0) := by
  /-
    α β : Type u
    ⊢ Iff (Eq (Cardinal.mk ↑(setOf fun f => Function.Surjective f)) 0) (Or (LT.lt  …
  -/
  rw [mk_surjective_eq_zero_iff_lift, lift_lt]
  /-
    🎉 no goals
  -/


theorem mk_equiv_le_embedding : #(α ≃ β') ≤ #(α ↪ β') := ⟨⟨_, Equiv.toEmbedding_injective⟩⟩


theorem mk_embedding_le_arrow : #(α ↪ β') ≤ #(α → β') := ⟨⟨_, DFunLike.coe_injective⟩⟩


theorem mk_perm_eq_self_power : #(Equiv.Perm α) = #α ^ #α :=
  ((mk_equiv_le_embedding α α).trans (mk_embedding_le_arrow α α)).antisymm <| by
    suffices Nonempty ((α → Bool) ↪ Equiv.Perm (α × Bool)) by
      obtain ⟨e⟩ : Nonempty (α ≃ α × Bool) := by
        erw [← Cardinal.eq, mk_prod, lift_uzero, mk_bool,
          lift_natCast, mul_two, add_eq_self (aleph0_le_mk α)]
      erw [← le_def, mk_arrow, lift_uzero, mk_bool, lift_natCast 2] at this
      rwa [← power_def, power_self_eq (aleph0_le_mk α), e.permCongr.cardinal_eq]
    /-
      α : Type u
      inst✝ : Infinite α
      ⊢ Nonempty (Function.Embedding (α → Bool) (Equiv.Perm (Prod α Bool)))
    -/
    refine ⟨⟨fun f ↦ Involutive.toPerm (fun x ↦ ⟨x.1, xor (f x.1) x.2⟩) fun x ↦ ?_, fun f g h ↦ ?_⟩⟩
      /-
        case refine_1
        α : Type u
        inst✝ : Infinite α
        f : α → Bool
        x : Prod α Bool
        ⊢ Eq ((fun x => { fst := x.1, snd := (f x.1).xor x.2 }) ((fun x => { fst := x. …
      -/
    · simp_rw [← Bool.xor_assoc, Bool.xor_self, Bool.false_xor]
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u
        inst✝ : Infinite α
        f g : α → Bool
        h : Eq ((fun f => Function.Involutive.toPerm (fun x => { fst := x.1, snd := (f …
        ⊢ Eq f g
      -/
    · ext a; rw [← (f a).xor_false, ← (g a).xor_false]; exact congr(($h ⟨a, false⟩).2)
                                                        /-
                                                          🎉 no goals
                                                        -/


theorem mk_perm_eq_two_power : #(Equiv.Perm α) = 2 ^ #α := by
  /-
    α : Type u
    inst✝ : Infinite α
    ⊢ Eq (Cardinal.mk (Equiv.Perm α)) (HPow.hPow 2 (Cardinal.mk α))
  -/
  rw [mk_perm_eq_self_power, power_self_eq (aleph0_le_mk α)]
  /-
    🎉 no goals
  -/


theorem mk_equiv_eq_arrow_of_lift_eq (leq : lift.{v} #α = lift.{u} #β') :
    #(α ≃ β') = #(α → β') := by
  /-
    α : Type u
    β' : Type v
    inst✝ : Infinite α
    leq : Eq (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardina …
    ⊢ Eq (Cardinal.mk (Equiv α β')) (Cardinal.mk (α → β'))
  -/
  obtain ⟨e⟩ := lift_mk_eq'.mp leq
  /-
    case intro
    α : Type u
    β' : Type v
    inst✝ : Infinite α
    leq : Eq (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardina …
    e : Equiv α β'
    ⊢ Eq (Cardinal.mk (Equiv α β')) (Cardinal.mk (α → β'))
  -/
  have e₁ := lift_mk_eq'.mpr ⟨.equivCongr (.refl α) e⟩
  /-
    case intro
    α : Type u
    β' : Type v
    inst✝ : Infinite α
    leq : Eq (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardina …
    e : Equiv α β'
    e₁ : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk (Equiv α α))) (Cardinal.lift. …
    ⊢ Eq (Cardinal.mk (Equiv α β')) (Cardinal.mk (α → β'))
  -/
  have e₂ := lift_mk_eq'.mpr ⟨.arrowCongr (.refl α) e⟩
  /-
    case intro
    α : Type u
    β' : Type v
    inst✝ : Infinite α
    leq : Eq (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardina …
    e : Equiv α β'
    e₁ : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk (Equiv α α))) (Cardinal.lift. …
    e₂ : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk (α → α))) (Cardinal.lift.{u,  …
    ⊢ Eq (Cardinal.mk (Equiv α β')) (Cardinal.mk (α → β'))
  -/
  rw [lift_id'.{u,v}] at e₁ e₂
  /-
    case intro
    α : Type u
    β' : Type v
    inst✝ : Infinite α
    leq : Eq (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardina …
    e : Equiv α β'
    e₁ : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk (Equiv α α))) (Cardinal.mk (E …
    e₂ : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk (α → α))) (Cardinal.mk (α → β …
    ⊢ Eq (Cardinal.mk (Equiv α β')) (Cardinal.mk (α → β'))
  -/
  rw [← e₁, ← e₂, lift_inj, mk_perm_eq_self_power, power_def]
  /-
    🎉 no goals
  -/


theorem mk_equiv_eq_arrow_of_eq (eq : #α = #β) : #(α ≃ β) = #(α → β) :=
  mk_equiv_eq_arrow_of_lift_eq congr(lift $eq)


theorem mk_equiv_of_lift_eq (leq : lift.{v} #α = lift.{u} #β') : #(α ≃ β') = 2 ^ lift.{v} #α := by
  erw [← (lift_mk_eq'.2 ⟨.equivCongr (.refl α) (lift_mk_eq'.1 leq).some⟩).trans (lift_id'.{u,v} _),
                                                                      /-
                                                                        α : Type u
                                                                        β' : Type v
                                                                        inst✝ : Infinite α
                                                                        leq : Eq (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} (Cardina …
                                                                        ⊢ Eq (HPow.hPow (↑2) (Cardinal.lift.{v, u} (Cardinal.mk α))) (HPow.hPow 2 (Car …
                                                                      -/
    lift_umax.{u,v}, mk_perm_eq_two_power, lift_power, lift_natCast]; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem mk_equiv_of_eq (eq : #α = #β) : #(α ≃ β) = 2 ^ #α := by
  /-
    α β : Type u
    inst✝ : Infinite α
    eq : Eq (Cardinal.mk α) (Cardinal.mk β)
    ⊢ Eq (Cardinal.mk (Equiv α β)) (HPow.hPow 2 (Cardinal.mk α))
  -/
  rw [mk_equiv_of_lift_eq (lift_inj.mpr eq), lift_id]
  /-
    🎉 no goals
  -/


theorem mk_embedding_eq_arrow_of_lift_le (lle : lift.{u} #β' ≤ lift.{v} #α) :
    #(β' ↪ α) = #(β' → α) :=
  (mk_embedding_le_arrow _ _).antisymm <| by
    conv_rhs => rw [← (Equiv.embeddingCongr (.refl _)
      (Cardinal.eq.mp <| mul_eq_self <| aleph0_le_mk α).some).cardinal_eq]
    /-
      α : Type u
      β' : Type v
      inst✝ : Infinite α
      lle : LE.le (Cardinal.lift.{u, v} (Cardinal.mk β')) (Cardinal.lift.{v, u} (Car …
      ⊢ LE.le (Cardinal.mk (β' → α)) (Cardinal.mk (Function.Embedding β' (Prod α α)))
    -/
    obtain ⟨e⟩ := lift_mk_le'.mp lle
    exact ⟨⟨fun f ↦ ⟨fun b ↦ ⟨e b, f b⟩, fun _ _ h ↦ e.injective congr(Prod.fst $h)⟩,
      fun f g h ↦ funext fun b ↦ congr(Prod.snd <| $h b)⟩⟩


theorem mk_embedding_eq_arrow_of_le (le : #β ≤ #α) : #(β ↪ α) = #(β → α) :=
  mk_embedding_eq_arrow_of_lift_le (lift_le.mpr le)


theorem mk_surjective_eq_arrow_of_lift_le (lle : lift.{u} #β' ≤ lift.{v} #α) :
    #{f : α → β' | Surjective f} = #(α → β') :=
  (mk_set_le _).antisymm <|
    have ⟨e⟩ : Nonempty (α ≃ α ⊕ β') := by
      /-
        α : Type u
        β' : Type v
        inst✝ : Infinite α
        lle : LE.le (Cardinal.lift.{u, v} (Cardinal.mk β')) (Cardinal.lift.{v, u} (Car …
        ⊢ Nonempty (Equiv α (Sum α β'))
      -/
      simp_rw [← lift_mk_eq', mk_sum, lift_add, lift_lift]; rw [lift_umax.{u,v}, eq_comm]
      /-
        α : Type u
        β' : Type v
        inst✝ : Infinite α
        lle : LE.le (Cardinal.lift.{u, v} (Cardinal.mk β')) (Cardinal.lift.{v, u} (Car …
        ⊢ Eq (HAdd.hAdd (Cardinal.lift.{v, u} (Cardinal.mk α)) (Cardinal.lift.{u, v} ( …
      -/
      exact add_eq_left (aleph0_le_lift.mpr <| aleph0_le_mk α) lle
      /-
        🎉 no goals
      -/
    ⟨⟨fun f ↦ ⟨fun a ↦ (e a).elim f id, fun b ↦ ⟨e.symm (.inr b), congr_arg _ (e.right_inv _)⟩⟩,
      fun f g h ↦ funext fun a ↦ by
        /-
          α : Type u
          β' : Type v
          inst✝ : Infinite α
          lle : LE.le (Cardinal.lift.{u, v} (Cardinal.mk β')) (Cardinal.lift.{v, u} (Car …
          e : Equiv α (Sum α β')
          f g : α → β'
          h : Eq ((fun f => ⟨fun a => Sum.elim f id (e a), ⋯⟩) f) ((fun f => ⟨fun a => S …
          a : α
          ⊢ Eq (f a) (g a)
        -/
        simpa only [e.apply_symm_apply] using congr_fun (Subtype.ext_iff.mp h) (e.symm <| .inl a)⟩⟩
        /-
          🎉 no goals
        -/


theorem mk_surjective_eq_arrow_of_le (le : #β ≤ #α) : #{f : α → β | Surjective f} = #(α → β) :=
  mk_surjective_eq_arrow_of_lift_le (lift_le.mpr le)


@[simp]
theorem mk_list_eq_mk (α : Type u) [Infinite α] : #(List α) = #α :=
  have H1 : ℵ₀ ≤ #α := aleph0_le_mk α
  Eq.symm <|
                                                            /-
                                                              α : Type u
                                                              inst✝ : Infinite α
                                                              H1 : LE.le Cardinal.aleph0 (Cardinal.mk α)
                                                              x✝ : α
                                                              ⊢ ∀ ⦃a₂ : α⦄, Eq ((fun a => List.cons a List.nil) x✝) ((fun a => List.cons a L …
                                                            -/
    le_antisymm ((le_def _ _).2 ⟨⟨fun a => [a], fun _ => by simp⟩⟩) <|
                                                            /-
                                                              🎉 no goals
                                                            -/
      calc
        #(List α) = sum fun n : ℕ => #α ^ (n : Cardinal.{u}) := mk_list_eq_sum_pow α
        _ ≤ sum fun _ : ℕ => #α := sum_le_sum _ _ fun n => pow_le H1 <| nat_lt_aleph0 n
                     /-
                       α : Type u
                       inst✝ : Infinite α
                       H1 : LE.le Cardinal.aleph0 (Cardinal.mk α)
                       ⊢ Eq (Cardinal.sum fun x => Cardinal.mk α) (Cardinal.mk α)
                     -/
        _ = #α := by simp [H1]
                     /-
                       🎉 no goals
                     -/


theorem mk_list_eq_aleph0 (α : Type u) [Countable α] [Nonempty α] : #(List α) = ℵ₀ :=
  mk_le_aleph0.antisymm (aleph0_le_mk _)


theorem mk_list_eq_max_mk_aleph0 (α : Type u) [Nonempty α] : #(List α) = max #α ℵ₀ := by
  /-
    α : Type u
    inst✝ : Nonempty α
    ⊢ Eq (Cardinal.mk (List α)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
  -/
  cases finite_or_infinite α
    /-
      case inl
      α : Type u
      inst✝ : Nonempty α
      h✝ : Finite α
      ⊢ Eq (Cardinal.mk (List α)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
    -/
  · rw [mk_list_eq_aleph0, eq_comm, max_eq_right]
    /-
      case inl
      α : Type u
      inst✝ : Nonempty α
      h✝ : Finite α
      ⊢ LE.le (Cardinal.mk α) Cardinal.aleph0
    -/
    exact mk_le_aleph0
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      inst✝ : Nonempty α
      h✝ : Infinite α
      ⊢ Eq (Cardinal.mk (List α)) (Max.max (Cardinal.mk α) Cardinal.aleph0)
    -/
  · rw [mk_list_eq_mk, eq_comm, max_eq_left]
    /-
      case inr
      α : Type u
      inst✝ : Nonempty α
      h✝ : Infinite α
      ⊢ LE.le Cardinal.aleph0 (Cardinal.mk α)
    -/
    exact aleph0_le_mk α
    /-
      🎉 no goals
    -/


theorem mk_list_le_max (α : Type u) : #(List α) ≤ max ℵ₀ #α := by
  /-
    α : Type u
    ⊢ LE.le (Cardinal.mk (List α)) (Max.max Cardinal.aleph0 (Cardinal.mk α))
  -/
  cases finite_or_infinite α
    /-
      case inl
      α : Type u
      h✝ : Finite α
      ⊢ LE.le (Cardinal.mk (List α)) (Max.max Cardinal.aleph0 (Cardinal.mk α))
    -/
  · exact mk_le_aleph0.trans (le_max_left _ _)
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u
      h✝ : Infinite α
      ⊢ LE.le (Cardinal.mk (List α)) (Max.max Cardinal.aleph0 (Cardinal.mk α))
    -/
  · rw [mk_list_eq_mk]
    /-
      case inr
      α : Type u
      h✝ : Infinite α
      ⊢ LE.le (Cardinal.mk α) (Max.max Cardinal.aleph0 (Cardinal.mk α))
    -/
    apply le_max_right
    /-
      🎉 no goals
    -/


@[simp]
theorem mk_finset_of_infinite (α : Type u) [Infinite α] : #(Finset α) = #α := by
  classical
  exact Eq.symm <|
    le_antisymm (mk_le_of_injective fun _ _ => Finset.singleton_inj.1) <|
      calc
        #(Finset α) ≤ #(List α) := mk_le_of_surjective List.toFinset_surjective
        _ = #α := mk_list_eq_mk α


theorem mk_bounded_set_le_of_infinite (α : Type u) [Infinite α] (c : Cardinal) :
    #{ t : Set α // #t ≤ c } ≤ #α ^ c := by
  /-
    α : Type u
    inst✝ : Infinite α
    c : Cardinal.{u}
    ⊢ LE.le (Cardinal.mk (Subtype fun t => LE.le (Cardinal.mk ↑t) c)) (HPow.hPow ( …
  -/
  refine le_trans ?_ (by rw [← add_one_eq (aleph0_le_mk α)])
  /-
    α : Type u
    inst✝ : Infinite α
    c : Cardinal.{u}
    ⊢ LE.le (Cardinal.mk (Subtype fun t => LE.le (Cardinal.mk ↑t) c)) (HPow.hPow ( …
  -/
  induction' c using Cardinal.inductionOn with β
  /-
    case h
    α : Type u
    inst✝ : Infinite α
    β : Type u
    ⊢ LE.le (Cardinal.mk (Subtype fun t => LE.le (Cardinal.mk ↑t) (Cardinal.mk β)) …
  -/
  fapply mk_le_of_surjective
    /-
      case h.f
      α : Type u
      inst✝ : Infinite α
      β : Type u
      ⊢ (fun α β => β → α) (Sum α (ULift.{u, 0} (Fin 1))) β → Subtype fun t => LE.le …
    -/
  · intro f
    /-
      case h.f
      α : Type u
      inst✝ : Infinite α
      β : Type u
      f : (fun α β => β → α) (Sum α (ULift.{u, 0} (Fin 1))) β
      ⊢ Subtype fun t => LE.le (Cardinal.mk ↑t) (Cardinal.mk β)
    -/
    use Sum.inl ⁻¹' range f
    /-
      case property
      α : Type u
      inst✝ : Infinite α
      β : Type u
      f : (fun α β => β → α) (Sum α (ULift.{u, 0} (Fin 1))) β
      ⊢ LE.le (Cardinal.mk ↑(Set.preimage Sum.inl (Set.range f))) (Cardinal.mk β)
    -/
    refine le_trans (mk_preimage_of_injective _ _ fun x y => Sum.inl.inj) ?_
    /-
      case property
      α : Type u
      inst✝ : Infinite α
      β : Type u
      f : (fun α β => β → α) (Sum α (ULift.{u, 0} (Fin 1))) β
      ⊢ LE.le (Cardinal.mk ↑(Set.range f)) (Cardinal.mk β)
    -/
    apply mk_range_le
    /-
      🎉 no goals
    -/
  /-
    case h.hf
    α : Type u
    inst✝ : Infinite α
    β : Type u
    ⊢ Function.Surjective fun f => ⟨Set.preimage Sum.inl (Set.range f), ⋯⟩
  -/
  rintro ⟨s, ⟨g⟩⟩
  classical
  use fun y => if h : ∃ x : s, g x = y then Sum.inl (Classical.choose h).val
               else Sum.inr (ULift.up 0)
  apply Subtype.eq; ext x
  constructor
  · rintro ⟨y, h⟩
    dsimp only at h
    by_cases h' : ∃ z : s, g z = y
    · rw [dif_pos h'] at h
      cases Sum.inl.inj h
      exact (Classical.choose h').2
    · rw [dif_neg h'] at h
      cases h
  · intro h
    have : ∃ z : s, g z = g ⟨x, h⟩ := ⟨⟨x, h⟩, rfl⟩
    use g ⟨x, h⟩
    dsimp only
    rw [dif_pos this]
    congr
    suffices Classical.choose this = ⟨x, h⟩ from congr_arg Subtype.val this
    apply g.2
    exact Classical.choose_spec this


theorem mk_bounded_set_le (α : Type u) (c : Cardinal) :
    #{ t : Set α // #t ≤ c } ≤ max #α ℵ₀ ^ c := by
  /-
    α : Type u
    c : Cardinal.{u}
    ⊢ LE.le (Cardinal.mk (Subtype fun t => LE.le (Cardinal.mk ↑t) c)) (HPow.hPow ( …
  -/
  trans #{ t : Set ((ULift.{u} ℕ) ⊕ α) // #t ≤ c }
    /-
      α : Type u
      c : Cardinal.{u}
      ⊢ LE.le (Cardinal.mk (Subtype fun t => LE.le (Cardinal.mk ↑t) c)) (Cardinal.mk …
    -/
  · refine ⟨Embedding.subtypeMap ?_ ?_⟩
      /-
        case refine_1
        α : Type u
        c : Cardinal.{u}
        ⊢ Function.Embedding (Set α) (Set (Sum (ULift.{u, 0} Nat) α))
      -/
    · apply Embedding.image
      /-
        case refine_1.f
        α : Type u
        c : Cardinal.{u}
        ⊢ Function.Embedding α (Sum (ULift.{u, 0} Nat) α)
      -/
      use Sum.inr
      /-
        case inj'
        α : Type u
        c : Cardinal.{u}
        ⊢ Function.Injective Sum.inr
      -/
      apply Sum.inr.inj
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      α : Type u
      c : Cardinal.{u}
      ⊢ ∀ ⦃x : Set α⦄, LE.le (Cardinal.mk ↑x) c → LE.le (Cardinal.mk ↑({ toFun := Su …
    -/
    intro s hs
    /-
      case refine_2
      α : Type u
      c : Cardinal.{u}
      s : Set α
      hs : LE.le (Cardinal.mk ↑s) c
      ⊢ LE.le (Cardinal.mk ↑({ toFun := Sum.inr, inj' := ⋯ }.image s)) c
    -/
    exact mk_image_le.trans hs
    /-
      🎉 no goals
    -/
  /-
    α : Type u
    c : Cardinal.{u}
    ⊢ LE.le (Cardinal.mk (Subtype fun t => LE.le (Cardinal.mk ↑t) c)) (HPow.hPow ( …
  -/
  apply (mk_bounded_set_le_of_infinite ((ULift.{u} ℕ) ⊕ α) c).trans
  /-
    α : Type u
    c : Cardinal.{u}
    ⊢ LE.le (HPow.hPow (Cardinal.mk (Sum (ULift.{u, 0} Nat) α)) c) (HPow.hPow (Max …
  -/
                                  /-
                                    🎉 no goals
                                  -/
  rw [max_comm, ← add_eq_max] <;> rfl
                                  /-
                                    🎉 no goals
                                  -/


theorem mk_bounded_subset_le {α : Type u} (s : Set α) (c : Cardinal.{u}) :
    #{ t : Set α // t ⊆ s ∧ #t ≤ c } ≤ max #s ℵ₀ ^ c := by
  /-
    α : Type u
    s : Set α
    c : Cardinal.{u}
    ⊢ LE.le (Cardinal.mk (Subtype fun t => And (HasSubset.Subset t s) (LE.le (Card …
  -/
  refine le_trans ?_ (mk_bounded_set_le s c)
  /-
    α : Type u
    s : Set α
    c : Cardinal.{u}
    ⊢ LE.le (Cardinal.mk (Subtype fun t => And (HasSubset.Subset t s) (LE.le (Card …
  -/
  refine ⟨Embedding.codRestrict _ ?_ ?_⟩
    /-
      case refine_1
      α : Type u
      s : Set α
      c : Cardinal.{u}
      ⊢ Function.Embedding (Subtype fun t => And (HasSubset.Subset t s) (LE.le (Card …
    -/
  · use fun t => (↑) ⁻¹' t.1
    /-
      case inj'
      α : Type u
      s : Set α
      c : Cardinal.{u}
      ⊢ Function.Injective fun t => Set.preimage Subtype.val ↑t
    -/
    rintro ⟨t, ht1, ht2⟩ ⟨t', h1t', h2t'⟩ h
    /-
      case inj'.mk.intro.mk.intro
      α : Type u
      s : Set α
      c : Cardinal.{u}
      t : Set α
      ht1 : HasSubset.Subset t s
      ht2 : LE.le (Cardinal.mk ↑t) c
      t' : Set α
      h1t' : HasSubset.Subset t' s
      h2t' : LE.le (Cardinal.mk ↑t') c
      h : Eq ((fun t => Set.preimage Subtype.val ↑t) ⟨t, ⋯⟩) ((fun t => Set.preimage …
      ⊢ Eq ⟨t, ⋯⟩ ⟨t', ⋯⟩
    -/
    apply Subtype.eq
    /-
      case inj'.mk.intro.mk.intro.a
      α : Type u
      s : Set α
      c : Cardinal.{u}
      t : Set α
      ht1 : HasSubset.Subset t s
      ht2 : LE.le (Cardinal.mk ↑t) c
      t' : Set α
      h1t' : HasSubset.Subset t' s
      h2t' : LE.le (Cardinal.mk ↑t') c
      h : Eq ((fun t => Set.preimage Subtype.val ↑t) ⟨t, ⋯⟩) ((fun t => Set.preimage …
      ⊢ Eq ↑⟨t, ⋯⟩ ↑⟨t', ⋯⟩
    -/
    dsimp only at h ⊢
    /-
      case inj'.mk.intro.mk.intro.a
      α : Type u
      s : Set α
      c : Cardinal.{u}
      t : Set α
      ht1 : HasSubset.Subset t s
      ht2 : LE.le (Cardinal.mk ↑t) c
      t' : Set α
      h1t' : HasSubset.Subset t' s
      h2t' : LE.le (Cardinal.mk ↑t') c
      h : Eq (Set.preimage Subtype.val t) (Set.preimage Subtype.val t')
      ⊢ Eq t t'
    -/
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
    refine (preimage_eq_preimage' ?_ ?_).1 h <;> rw [Subtype.range_coe] <;> assumption
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  /-
    case refine_2
    α : Type u
    s : Set α
    c : Cardinal.{u}
    ⊢ ∀ (a : Subtype fun t => And (HasSubset.Subset t s) (LE.le (Cardinal.mk ↑t) c …
  -/
  rintro ⟨t, _, h2t⟩; exact (mk_preimage_of_injective _ _ Subtype.val_injective).trans h2t
                      /-
                        🎉 no goals
                      -/


theorem mk_compl_of_infinite {α : Type*} [Infinite α] (s : Set α) (h2 : #s < #α) :
    #(sᶜ : Set α) = #α := by
  /-
    α : Type u_1
    inst✝ : Infinite α
    s : Set α
    h2 : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
    ⊢ Eq (Cardinal.mk ↑(HasCompl.compl s)) (Cardinal.mk α)
  -/
  refine eq_of_add_eq_of_aleph0_le ?_ h2 (aleph0_le_mk α)
  /-
    α : Type u_1
    inst✝ : Infinite α
    s : Set α
    h2 : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
    ⊢ Eq (HAdd.hAdd (Cardinal.mk ↑s) (Cardinal.mk ↑(HasCompl.compl s))) (Cardinal. …
  -/
  exact mk_sum_compl s
  /-
    🎉 no goals
  -/


theorem mk_compl_finset_of_infinite {α : Type*} [Infinite α] (s : Finset α) :
    #((↑s)ᶜ : Set α) = #α := by
  /-
    α : Type u_1
    inst✝ : Infinite α
    s : Finset α
    ⊢ Eq (Cardinal.mk ↑(HasCompl.compl ↑s)) (Cardinal.mk α)
  -/
  apply mk_compl_of_infinite
  /-
    case h2
    α : Type u_1
    inst✝ : Infinite α
    s : Finset α
    ⊢ LT.lt (Cardinal.mk ↑↑s) (Cardinal.mk α)
  -/
  exact (finset_card_lt_aleph0 s).trans_le (aleph0_le_mk α)
  /-
    🎉 no goals
  -/


theorem mk_compl_eq_mk_compl_infinite {α : Type*} [Infinite α] {s t : Set α} (hs : #s < #α)
    (ht : #t < #α) : #(sᶜ : Set α) = #(tᶜ : Set α) := by
  /-
    α : Type u_1
    inst✝ : Infinite α
    s t : Set α
    hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
    ht : LT.lt (Cardinal.mk ↑t) (Cardinal.mk α)
    ⊢ Eq (Cardinal.mk ↑(HasCompl.compl s)) (Cardinal.mk ↑(HasCompl.compl t))
  -/
  rw [mk_compl_of_infinite s hs, mk_compl_of_infinite t ht]
  /-
    🎉 no goals
  -/


theorem mk_compl_eq_mk_compl_finite_lift {α : Type u} {β : Type v} [Finite α] {s : Set α}
    {t : Set β} (h1 : (lift.{max v w, u} #α) = (lift.{max u w, v} #β))
    (h2 : lift.{max v w, u} #s = lift.{max u w, v} #t) :
    lift.{max v w} #(sᶜ : Set α) = lift.{max u w} #(tᶜ : Set β) := by
  /-
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    t : Set β
    h1 : Eq (Cardinal.lift.{max v w, u} (Cardinal.mk α)) (Cardinal.lift.{max u w,  …
    h2 : Eq (Cardinal.lift.{max v w, u} (Cardinal.mk ↑s)) (Cardinal.lift.{max u w, …
    ⊢ Eq (Cardinal.lift.{max v w, u} (Cardinal.mk ↑(HasCompl.compl s))) (Cardinal. …
  -/
  cases nonempty_fintype α
  /-
    case intro
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    t : Set β
    h1 : Eq (Cardinal.lift.{max v w, u} (Cardinal.mk α)) (Cardinal.lift.{max u w,  …
    h2 : Eq (Cardinal.lift.{max v w, u} (Cardinal.mk ↑s)) (Cardinal.lift.{max u w, …
    val✝ : Fintype α
    ⊢ Eq (Cardinal.lift.{max v w, u} (Cardinal.mk ↑(HasCompl.compl s))) (Cardinal. …
  -/
  rcases lift_mk_eq.{u, v, w}.1 h1 with ⟨e⟩; letI : Fintype β := Fintype.ofEquiv α e
  /-
    case intro.intro
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    t : Set β
    h1 : Eq (Cardinal.lift.{max v w, u} (Cardinal.mk α)) (Cardinal.lift.{max u w,  …
    h2 : Eq (Cardinal.lift.{max v w, u} (Cardinal.mk ↑s)) (Cardinal.lift.{max u w, …
    val✝ : Fintype α
    e : Equiv α β
    this : Fintype β := Fintype.ofEquiv α e
    ⊢ Eq (Cardinal.lift.{max v w, u} (Cardinal.mk ↑(HasCompl.compl s))) (Cardinal. …
  -/
  replace h1 : Fintype.card α = Fintype.card β := (Fintype.ofEquiv_card _).symm
  classical
    lift s to Finset α using s.toFinite
    lift t to Finset β using t.toFinite
    simp only [Finset.coe_sort_coe, mk_fintype, Fintype.card_coe, lift_natCast, Nat.cast_inj] at h2
    simp only [← Finset.coe_compl, Finset.coe_sort_coe, mk_coe_finset, Finset.card_compl,
      lift_natCast, Nat.cast_inj, h1, h2]


theorem mk_compl_eq_mk_compl_finite {α β : Type u} [Finite α] {s : Set α} {t : Set β}
    (h1 : #α = #β) (h : #s = #t) : #(sᶜ : Set α) = #(tᶜ : Set β) := by
  /-
    α β : Type u
    inst✝ : Finite α
    s : Set α
    t : Set β
    h1 : Eq (Cardinal.mk α) (Cardinal.mk β)
    h : Eq (Cardinal.mk ↑s) (Cardinal.mk ↑t)
    ⊢ Eq (Cardinal.mk ↑(HasCompl.compl s)) (Cardinal.mk ↑(HasCompl.compl t))
  -/
  rw [← lift_inj.{u, u}]
  /-
    α β : Type u
    inst✝ : Finite α
    s : Set α
    t : Set β
    h1 : Eq (Cardinal.mk α) (Cardinal.mk β)
    h : Eq (Cardinal.mk ↑s) (Cardinal.mk ↑t)
    ⊢ Eq (Cardinal.lift.{u, u} (Cardinal.mk ↑(HasCompl.compl s))) (Cardinal.lift.{ …
  -/
  apply mk_compl_eq_mk_compl_finite_lift.{u, u, u}
      /-
        case h1
        α β : Type u
        inst✝ : Finite α
        s : Set α
        t : Set β
        h1 : Eq (Cardinal.mk α) (Cardinal.mk β)
        h : Eq (Cardinal.mk ↑s) (Cardinal.mk ↑t)
        ⊢ Eq (Cardinal.lift.{u, u} (Cardinal.mk α)) (Cardinal.lift.{u, u} (Cardinal.mk …
      -/
      /-
        🎉 no goals
      -/
  <;> rwa [lift_inj]
      /-
        🎉 no goals
      -/


theorem mk_compl_eq_mk_compl_finite_same {α : Type u} [Finite α] {s t : Set α} (h : #s = #t) :
    #(sᶜ : Set α) = #(tᶜ : Set α) :=
  mk_compl_eq_mk_compl_finite.{u} rfl h


theorem extend_function {α β : Type*} {s : Set α} (f : s ↪ β)
    (h : Nonempty ((sᶜ : Set α) ≃ ((range f)ᶜ : Set β))) : ∃ g : α ≃ β, ∀ x : s, g x = f x := by
  classical
  have := h; cases' this with g
  let h : α ≃ β :=
    (Set.sumCompl (s : Set α)).symm.trans
      ((sumCongr (Equiv.ofInjective f f.2) g).trans (Set.sumCompl (range f)))
  refine ⟨h, ?_⟩; rintro ⟨x, hx⟩; simp [h, Set.sumCompl_symm_apply_of_mem, hx]


theorem extend_function_finite {α : Type u} {β : Type v} [Finite α] {s : Set α} (f : s ↪ β)
    (h : Nonempty (α ≃ β)) : ∃ g : α ≃ β, ∀ x : s, g x = f x := by
  /-
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    f : Function.Embedding (↑s) β
    h : Nonempty (Equiv α β)
    ⊢ Exists fun g => ∀ (x : ↑s), Eq (g ↑x) (f x)
  -/
  apply extend_function.{u, v} f
  /-
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    f : Function.Embedding (↑s) β
    h : Nonempty (Equiv α β)
    ⊢ Nonempty (Equiv ↑(HasCompl.compl s) ↑(HasCompl.compl (Set.range ⇑f)))
  -/
  cases' id h with g
  /-
    case intro
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    f : Function.Embedding (↑s) β
    h : Nonempty (Equiv α β)
    g : Equiv α β
    ⊢ Nonempty (Equiv ↑(HasCompl.compl s) ↑(HasCompl.compl (Set.range ⇑f)))
  -/
  rw [← lift_mk_eq.{u, v, max u v}] at h
  /-
    case intro
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    f : Function.Embedding (↑s) β
    h : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk α)) (Cardinal.lift.{max u v, v …
    g : Equiv α β
    ⊢ Nonempty (Equiv ↑(HasCompl.compl s) ↑(HasCompl.compl (Set.range ⇑f)))
  -/
  rw [← lift_mk_eq.{u, v, max u v}, mk_compl_eq_mk_compl_finite_lift.{u, v, max u v} h]
  /-
    case intro
    α : Type u
    β : Type v
    inst✝ : Finite α
    s : Set α
    f : Function.Embedding (↑s) β
    h : Eq (Cardinal.lift.{max u v, u} (Cardinal.mk α)) (Cardinal.lift.{max u v, v …
    g : Equiv α β
    ⊢ Eq (Cardinal.lift.{max u v, u} (Cardinal.mk ↑s)) (Cardinal.lift.{max u v, v} …
  -/
  rw [mk_range_eq_lift.{u, v, max u v}]; exact f.2
                                         /-
                                           🎉 no goals
                                         -/


theorem extend_function_of_lt {α β : Type*} {s : Set α} (f : s ↪ β) (hs : #s < #α)
    (h : Nonempty (α ≃ β)) : ∃ g : α ≃ β, ∀ x : s, g x = f x := by
  /-
    α : Type u_1
    β : Type u_2
    s : Set α
    f : Function.Embedding (↑s) β
    hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
    h : Nonempty (Equiv α β)
    ⊢ Exists fun g => ∀ (x : ↑s), Eq (g ↑x) (f x)
  -/
  cases fintypeOrInfinite α
    /-
      case inl
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Nonempty (Equiv α β)
      val✝ : Fintype α
      ⊢ Exists fun g => ∀ (x : ↑s), Eq (g ↑x) (f x)
    -/
  · exact extend_function_finite f h
    /-
      🎉 no goals
    -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Nonempty (Equiv α β)
      val✝ : Infinite α
      ⊢ Exists fun g => ∀ (x : ↑s), Eq (g ↑x) (f x)
    -/
  · apply extend_function f
    /-
      case inr
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Nonempty (Equiv α β)
      val✝ : Infinite α
      ⊢ Nonempty (Equiv ↑(HasCompl.compl s) ↑(HasCompl.compl (Set.range ⇑f)))
    -/
    cases' id h with g
    /-
      case inr.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Nonempty (Equiv α β)
      val✝ : Infinite α
      g : Equiv α β
      ⊢ Nonempty (Equiv ↑(HasCompl.compl s) ↑(HasCompl.compl (Set.range ⇑f)))
    -/
    haveI := Infinite.of_injective _ g.injective
    /-
      case inr.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Nonempty (Equiv α β)
      val✝ : Infinite α
      g : Equiv α β
      this : Infinite β
      ⊢ Nonempty (Equiv ↑(HasCompl.compl s) ↑(HasCompl.compl (Set.range ⇑f)))
    -/
    rw [← lift_mk_eq'] at h ⊢
    /-
      case inr.intro
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Eq (Cardinal.lift.{u_2, u_1} (Cardinal.mk α)) (Cardinal.lift.{u_1, u_2} (C …
      val✝ : Infinite α
      g : Equiv α β
      this : Infinite β
      ⊢ Eq (Cardinal.lift.{u_2, u_1} (Cardinal.mk ↑(HasCompl.compl s))) (Cardinal.li …
    -/
    rwa [mk_compl_of_infinite s hs, mk_compl_of_infinite]
    /-
      case inr.intro.h2
      α : Type u_1
      β : Type u_2
      s : Set α
      f : Function.Embedding (↑s) β
      hs : LT.lt (Cardinal.mk ↑s) (Cardinal.mk α)
      h : Eq (Cardinal.lift.{u_2, u_1} (Cardinal.mk α)) (Cardinal.lift.{u_1, u_2} (C …
      val✝ : Infinite α
      g : Equiv α β
      this : Infinite β
      ⊢ LT.lt (Cardinal.mk ↑(Set.range ⇑f)) (Cardinal.mk β)
    -/
    rwa [← lift_lt, mk_range_eq_of_injective f.injective, ← h, lift_lt]
    /-
      🎉 no goals
    -/


/-- Bounds the cardinal of an ordinal-indexed union of sets. -/
lemma mk_iUnion_Ordinal_lift_le_of_le {β : Type v} {o : Ordinal.{u}} {c : Cardinal.{v}}
    (ho : lift.{v} o.card ≤ lift.{u} c) (hc : ℵ₀ ≤ c) (A : Ordinal → Set β)
    (hA : ∀ j < o, #(A j) ≤ c) : #(⋃ j < o, A j) ≤ c := by
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (Cardinal.mk ↑(Set.iUnion fun j => Set.iUnion fun h => A j)) c
  -/
  simp_rw [← mem_Iio, biUnion_eq_iUnion, iUnion, iSup, ← o.enumIsoToType.symm.surjective.range_comp]
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (Cardinal.mk ↑(SupSet.sSup (Set.range (Function.comp (fun x => A ↑x) ⇑ …
  -/
  rw [← lift_le.{u}]
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(SupSet.sSup (Set.range (Function. …
  -/
  apply ((mk_iUnion_le_lift _).trans _).trans_eq (mul_eq_self (aleph0_le_lift.2 hc))
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (HMul.hMul (Cardinal.lift.{v, u} (Cardinal.mk o.toType)) (iSup fun i = …
  -/
  rw [mk_toType]
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (HMul.hMul (Cardinal.lift.{v, u} o.card) (iSup fun i => Cardinal.lift. …
  -/
  refine mul_le_mul' ho (ciSup_le' ?_)
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ ∀ (i : o.toType), LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Function.comp ( …
  -/
  intro i
  /-
    β : Type v
    o : Ordinal.{u}
    c : Cardinal.{v}
    ho : LE.le (Cardinal.lift.{v, u} o.card) (Cardinal.lift.{u, v} c)
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u} → Set β
    hA : ∀ (j : Ordinal.{u}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    i : o.toType
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk ↑(Function.comp (fun x => A ↑x) (⇑o …
  -/
  simpa using hA _ (o.enumIsoToType.symm i).2
  /-
    🎉 no goals
  -/


lemma mk_iUnion_Ordinal_le_of_le {β : Type*} {o : Ordinal} {c : Cardinal}
    (ho : o.card ≤ c) (hc : ℵ₀ ≤ c) (A : Ordinal → Set β)
    (hA : ∀ j < o, #(A j) ≤ c) : #(⋃ j < o, A j) ≤ c := by
  /-
    β : Type u_1
    o : Ordinal.{u_1}
    c : Cardinal.{u_1}
    ho : LE.le o.card c
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u_1} → Set β
    hA : ∀ (j : Ordinal.{u_1}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (Cardinal.mk ↑(Set.iUnion fun j => Set.iUnion fun h => A j)) c
  -/
  apply mk_iUnion_Ordinal_lift_le_of_le _ hc A hA
  /-
    β : Type u_1
    o : Ordinal.{u_1}
    c : Cardinal.{u_1}
    ho : LE.le o.card c
    hc : LE.le Cardinal.aleph0 c
    A : Ordinal.{u_1} → Set β
    hA : ∀ (j : Ordinal.{u_1}), LT.lt j o → LE.le (Cardinal.mk ↑(A j)) c
    ⊢ LE.le (Cardinal.lift.{u_1, u_1} o.card) (Cardinal.lift.{u_1, u_1} c)
  -/
  rwa [Cardinal.lift_le]
  /-
    🎉 no goals
  -/


@[deprecated mk_iUnion_Ordinal_le_of_le (since := "2024-11-02")]
alias Ordinal.Cardinal.mk_iUnion_Ordinal_le_of_le := mk_iUnion_Ordinal_le_of_le


theorem lift_card_iSup_le_sum_card {ι : Type u} [Small.{v} ι] (f : ι → Ordinal.{v}) :
    Cardinal.lift.{u} (⨆ i, f i).card ≤ Cardinal.sum fun i ↦ (f i).card := by
  /-
    ι : Type u
    inst✝ : Small.{v, u} ι
    f : ι → Ordinal.{v}
    ⊢ LE.le (Cardinal.lift.{u, v} (iSup fun i => f i).card) (Cardinal.sum fun i => …
  -/
  simp_rw [← mk_toType]
  /-
    ι : Type u
    inst✝ : Small.{v, u} ι
    f : ι → Ordinal.{v}
    ⊢ LE.le (Cardinal.lift.{u, v} (Cardinal.mk (iSup fun i => f i).toType)) (Cardi …
  -/
  rw [← mk_sigma, ← Cardinal.lift_id'.{v} #(Σ _, _), ← Cardinal.lift_umax.{v, u}]
  apply lift_mk_le_lift_mk_of_surjective (f := enumIsoToType _ ∘ (⟨(enumIsoToType _).symm ·.2,
    (mem_Iio.mp ((enumIsoToType _).symm _).2).trans_le (Ordinal.le_iSup _ _)⟩))
  /-
    ι : Type u
    inst✝ : Small.{v, u} ι
    f : ι → Ordinal.{v}
    ⊢ Function.Surjective (Function.comp ⇑(iSup fun i => f i).enumIsoToType fun x  …
  -/
  rw [EquivLike.comp_surjective]
  /-
    ι : Type u
    inst✝ : Small.{v, u} ι
    f : ι → Ordinal.{v}
    ⊢ Function.Surjective fun x => ⟨↑((f x.fst).enumIsoToType.symm x.snd), ⋯⟩
  -/
  rintro ⟨x, hx⟩
  /-
    case mk
    ι : Type u
    inst✝ : Small.{v, u} ι
    f : ι → Ordinal.{v}
    x : Ordinal.{v}
    hx : Membership.mem (Set.Iio (iSup fun i => f i)) x
    ⊢ Exists fun a => Eq ((fun x => ⟨↑((f x.fst).enumIsoToType.symm x.snd), ⋯⟩) a) …
  -/
  obtain ⟨i, hi⟩ := Ordinal.lt_iSup_iff.mp hx
  /-
    case mk.intro
    ι : Type u
    inst✝ : Small.{v, u} ι
    f : ι → Ordinal.{v}
    x : Ordinal.{v}
    hx : Membership.mem (Set.Iio (iSup fun i => f i)) x
    i : ι
    hi : LT.lt x (f i)
    ⊢ Exists fun a => Eq ((fun x => ⟨↑((f x.fst).enumIsoToType.symm x.snd), ⋯⟩) a) …
  -/
  exact ⟨⟨i, enumIsoToType _ ⟨x, hi⟩⟩, by simp⟩
  /-
    🎉 no goals
  -/


theorem card_iSup_le_sum_card {ι : Type u} (f : ι → Ordinal.{max u v}) :
    (⨆ i, f i).card ≤ Cardinal.sum (fun i ↦ (f i).card) := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (iSup fun i => f i).card (Cardinal.sum fun i => (f i).card)
  -/
  have := lift_card_iSup_le_sum_card f
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    this : LE.le (Cardinal.lift.{u, max u v} (iSup fun i => f i).card) (Cardinal.s …
    ⊢ LE.le (iSup fun i => f i).card (Cardinal.sum fun i => (f i).card)
  -/
  rwa [Cardinal.lift_id'] at this
  /-
    🎉 no goals
  -/


theorem card_iSup_Iio_le_sum_card {o : Ordinal.{u}} (f : Iio o → Ordinal.{max u v}) :
    (⨆ a : Iio o, f a).card ≤ Cardinal.sum fun i ↦ (f ((enumIsoToType o).symm i)).card := by
  /-
    o : Ordinal.{u}
    f : ↑(Set.Iio o) → Ordinal.{max u v}
    ⊢ LE.le (iSup fun a => f a).card (Cardinal.sum fun i => (f (o.enumIsoToType.sy …
  -/
  apply le_of_eq_of_le (congr_arg _ _).symm (card_iSup_le_sum_card _)
  /-
    o : Ordinal.{u}
    f : ↑(Set.Iio o) → Ordinal.{max u v}
    ⊢ Eq (iSup fun i => f (o.enumIsoToType.symm i)) (iSup fun a => f a)
  -/
  simpa using (enumIsoToType o).symm.iSup_comp (g := fun x ↦ f x)
  /-
    🎉 no goals
  -/


theorem card_iSup_Iio_le_card_mul_iSup {o : Ordinal.{u}} (f : Iio o → Ordinal.{max u v}) :
    (⨆ a : Iio o, f a).card ≤ Cardinal.lift.{v} o.card * ⨆ a : Iio o, (f a).card := by
  /-
    o : Ordinal.{u}
    f : ↑(Set.Iio o) → Ordinal.{max u v}
    ⊢ LE.le (iSup fun a => f a).card (HMul.hMul (Cardinal.lift.{v, u} o.card) (iSu …
  -/
  apply (card_iSup_Iio_le_sum_card f).trans
  /-
    o : Ordinal.{u}
    f : ↑(Set.Iio o) → Ordinal.{max u v}
    ⊢ LE.le (Cardinal.sum fun i => (f (o.enumIsoToType.symm i)).card) (HMul.hMul ( …
  -/
  convert ← sum_le_iSup_lift _
    /-
      case h.e'_4.h.e'_5.h.e'_1
      o : Ordinal.{u}
      f : ↑(Set.Iio o) → Ordinal.{max u v}
      ⊢ Eq (Cardinal.mk o.toType) o.card
    -/
  · exact mk_toType o
    /-
      🎉 no goals
    -/
    /-
      case h.e'_4.h.e'_6
      o : Ordinal.{u}
      f : ↑(Set.Iio o) → Ordinal.{max u v}
      ⊢ Eq (iSup fun i => (f (o.enumIsoToType.symm i)).card) (iSup fun a => (f a).ca …
    -/
  · exact (enumIsoToType o).symm.iSup_comp (g := fun x ↦ (f x).card)
    /-
      🎉 no goals
    -/


theorem card_opow_le_of_omega0_le_left {a : Ordinal} (ha : ω ≤ a) (b : Ordinal) :
    (a ^ b).card ≤ max a.card b.card := by
  /-
    a : Ordinal.{u_1}
    ha : LE.le Ordinal.omega0 a
    b : Ordinal.{u_1}
    ⊢ LE.le (HPow.hPow a b).card (Max.max a.card b.card)
  -/
  refine limitRecOn b ?_ ?_ ?_
    /-
      case refine_1
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b : Ordinal.{u_1}
      ⊢ LE.le (HPow.hPow a 0).card (Max.max a.card (Ordinal.card 0))
    -/
  · simpa using one_lt_omega0.le.trans ha
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b : Ordinal.{u_1}
      ⊢ ∀ (o : Ordinal.{u_1}), LE.le (HPow.hPow a o).card (Max.max a.card o.card) →  …
    -/
  · intro b IH
    /-
      case refine_2
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b✝ b : Ordinal.{u_1}
      IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
      ⊢ LE.le (HPow.hPow a (Order.succ b)).card (Max.max a.card (Order.succ b).card)
    -/
    rw [opow_succ, card_mul, card_succ, Cardinal.mul_eq_max_of_aleph0_le_right, max_comm]
      /-
        case refine_2
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
        ⊢ LE.le (Max.max a.card (HPow.hPow a b).card) (Max.max a.card (HAdd.hAdd b.car …
      -/
    · apply (max_le_max_left _ IH).trans
      /-
        case refine_2
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
        ⊢ LE.le (Max.max a.card (Max.max a.card b.card)) (Max.max a.card (HAdd.hAdd b. …
      -/
      rw [← max_assoc, max_self]
      /-
        case refine_2
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
        ⊢ LE.le (Max.max a.card b.card) (Max.max a.card (HAdd.hAdd b.card 1))
      -/
      exact max_le_max_left _ le_self_add
      /-
        🎉 no goals
      -/
      /-
        case refine_2.h'
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
        ⊢ Ne (HPow.hPow a b).card 0
      -/
    · rw [ne_eq, card_eq_zero, opow_eq_zero]
      /-
        case refine_2.h'
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
        ⊢ Not (And (Eq a 0) (Ne b 0))
      -/
      rintro ⟨rfl, -⟩
      /-
        case refine_2.h'.intro
        b✝ b : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 0
        IH : LE.le (HPow.hPow 0 b).card (Max.max (Ordinal.card 0) b.card)
        ⊢ False
      -/
      cases omega0_pos.not_le ha
      /-
        🎉 no goals
      -/
      /-
        case refine_2.h
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        IH : LE.le (HPow.hPow a b).card (Max.max a.card b.card)
        ⊢ LE.le Cardinal.aleph0 a.card
      -/
    · rwa [aleph0_le_card]
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b : Ordinal.{u_1}
      ⊢ ∀ (o : Ordinal.{u_1}), o.IsLimit → (∀ (o' : Ordinal.{u_1}), LT.lt o' o → LE. …
    -/
  · intro b hb IH
    /-
      case refine_3
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b✝ b : Ordinal.{u_1}
      hb : b.IsLimit
      IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
      ⊢ LE.le (HPow.hPow a b).card (Max.max a.card b.card)
    -/
    rw [(isNormal_opow (one_lt_omega0.trans_le ha)).apply_of_isLimit hb]
    /-
      case refine_3
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b✝ b : Ordinal.{u_1}
      hb : b.IsLimit
      IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
      ⊢ LE.le (iSup fun a_1 => HPow.hPow a ↑a_1).card (Max.max a.card b.card)
    -/
    apply (card_iSup_Iio_le_card_mul_iSup _).trans
    /-
      case refine_3
      a : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      b✝ b : Ordinal.{u_1}
      hb : b.IsLimit
      IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
      ⊢ LE.le (HMul.hMul (Cardinal.lift.{u_1, u_1} b.card) (iSup fun a_1 => (HPow.hP …
    -/
    rw [Cardinal.lift_id, Cardinal.mul_eq_max_of_aleph0_le_right, max_comm]
      /-
        case refine_3
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        hb : b.IsLimit
        IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
        ⊢ LE.le (Max.max (iSup fun a_1 => (HPow.hPow a ↑a_1).card) b.card) (Max.max a. …
      -/
    · apply max_le _ (le_max_right _ _)
      /-
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        hb : b.IsLimit
        IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
        ⊢ LE.le (iSup fun a_1 => (HPow.hPow a ↑a_1).card) (Max.max a.card b.card)
      -/
      apply ciSup_le'
      /-
        case h
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        hb : b.IsLimit
        IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
        ⊢ ∀ (i : ↑(Set.Iio b)), LE.le (HPow.hPow a ↑i).card (Max.max a.card b.card)
      -/
      intro c
      /-
        case h
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        hb : b.IsLimit
        IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
        c : ↑(Set.Iio b)
        ⊢ LE.le (HPow.hPow a ↑c).card (Max.max a.card b.card)
      -/
      exact (IH c.1 c.2).trans (max_le_max_left _ (card_le_card c.2.le))
      /-
        🎉 no goals
      -/
      /-
        case refine_3.h'
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        hb : b.IsLimit
        IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
        ⊢ Ne b.card 0
      -/
    · simpa using hb.pos.ne'
      /-
        🎉 no goals
      -/
      /-
        case refine_3.h
        a : Ordinal.{u_1}
        ha : LE.le Ordinal.omega0 a
        b✝ b : Ordinal.{u_1}
        hb : b.IsLimit
        IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
        ⊢ LE.le Cardinal.aleph0 (iSup fun a_1 => (HPow.hPow a ↑a_1).card)
      -/
    · refine le_ciSup_of_le ?_ ⟨1, one_lt_omega0.trans_le <| omega0_le_of_isLimit hb⟩ ?_
        /-
          case refine_3.h.refine_1
          a : Ordinal.{u_1}
          ha : LE.le Ordinal.omega0 a
          b✝ b : Ordinal.{u_1}
          hb : b.IsLimit
          IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
          ⊢ BddAbove (Set.range fun a_1 => (HPow.hPow a ↑a_1).card)
        -/
      · exact Cardinal.bddAbove_of_small _
        /-
          🎉 no goals
        -/
        /-
          case refine_3.h.refine_2
          a : Ordinal.{u_1}
          ha : LE.le Ordinal.omega0 a
          b✝ b : Ordinal.{u_1}
          hb : b.IsLimit
          IH : ∀ (o' : Ordinal.{u_1}), LT.lt o' b → LE.le (HPow.hPow a o').card (Max.max …
          ⊢ LE.le Cardinal.aleph0 (HPow.hPow a ↑⟨1, ⋯⟩).card
        -/
      · simpa
        /-
          🎉 no goals
        -/


theorem card_opow_le_of_omega0_le_right (a : Ordinal) {b : Ordinal} (hb : ω ≤ b) :
    (a ^ b).card ≤ max a.card b.card := by
  /-
    a b : Ordinal.{u_1}
    hb : LE.le Ordinal.omega0 b
    ⊢ LE.le (HPow.hPow a b).card (Max.max a.card b.card)
  -/
  obtain ⟨n, rfl⟩ | ha := eq_nat_or_omega0_le a
    /-
      case inl.intro
      b : Ordinal.{u_1}
      hb : LE.le Ordinal.omega0 b
      n : Nat
      ⊢ LE.le (HPow.hPow (↑n) b).card (Max.max (↑n).card b.card)
    -/
  · apply (card_le_card <| opow_le_opow_left b (nat_lt_omega0 n).le).trans
    /-
      case inl.intro
      b : Ordinal.{u_1}
      hb : LE.le Ordinal.omega0 b
      n : Nat
      ⊢ LE.le (HPow.hPow Ordinal.omega0 b).card (Max.max (↑n).card b.card)
    -/
    apply (card_opow_le_of_omega0_le_left le_rfl _).trans
    /-
      case inl.intro
      b : Ordinal.{u_1}
      hb : LE.le Ordinal.omega0 b
      n : Nat
      ⊢ LE.le (Max.max Ordinal.omega0.card b.card) (Max.max (↑n).card b.card)
    -/
    simp [hb]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Ordinal.{u_1}
      hb : LE.le Ordinal.omega0 b
      ha : LE.le Ordinal.omega0 a
      ⊢ LE.le (HPow.hPow a b).card (Max.max a.card b.card)
    -/
  · exact card_opow_le_of_omega0_le_left ha b
    /-
      🎉 no goals
    -/


theorem card_opow_le (a b : Ordinal) : (a ^ b).card ≤ max ℵ₀ (max a.card b.card) := by
  /-
    a b : Ordinal.{u_1}
    ⊢ LE.le (HPow.hPow a b).card (Max.max Cardinal.aleph0 (Max.max a.card b.card))
  -/
  obtain ⟨n, rfl⟩ | ha := eq_nat_or_omega0_le a
    /-
      case inl.intro
      b : Ordinal.{u_1}
      n : Nat
      ⊢ LE.le (HPow.hPow (↑n) b).card (Max.max Cardinal.aleph0 (Max.max (↑n).card b. …
    -/
  · obtain ⟨m, rfl⟩ | hb := eq_nat_or_omega0_le b
      /-
        case inl.intro.inl.intro
        n m : Nat
        ⊢ LE.le (HPow.hPow ↑n ↑m).card (Max.max Cardinal.aleph0 (Max.max (↑n).card (↑m …
      -/
    · rw [← natCast_opow, card_nat]
      /-
        case inl.intro.inl.intro
        n m : Nat
        ⊢ LE.le (↑(HPow.hPow n m)) (Max.max Cardinal.aleph0 (Max.max (↑n).card (↑m).ca …
      -/
      exact le_max_of_le_left (nat_lt_aleph0 _).le
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.inr
        b : Ordinal.{u_1}
        n : Nat
        hb : LE.le Ordinal.omega0 b
        ⊢ LE.le (HPow.hPow (↑n) b).card (Max.max Cardinal.aleph0 (Max.max (↑n).card b. …
      -/
    · exact (card_opow_le_of_omega0_le_right _ hb).trans (le_max_right _ _)
      /-
        🎉 no goals
      -/
    /-
      case inr
      a b : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      ⊢ LE.le (HPow.hPow a b).card (Max.max Cardinal.aleph0 (Max.max a.card b.card))
    -/
  · exact (card_opow_le_of_omega0_le_left ha _).trans (le_max_right _ _)
    /-
      🎉 no goals
    -/


theorem card_opow_eq_of_omega0_le_left {a b : Ordinal} (ha : ω ≤ a) (hb : 0 < b) :
    (a ^ b).card = max a.card b.card := by
  /-
    a b : Ordinal.{u_1}
    ha : LE.le Ordinal.omega0 a
    hb : LT.lt 0 b
    ⊢ Eq (HPow.hPow a b).card (Max.max a.card b.card)
  -/
  apply (card_opow_le_of_omega0_le_left ha b).antisymm (max_le _ _) <;> apply card_le_card
    /-
      case a
      a b : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      hb : LT.lt 0 b
      ⊢ LE.le a (HPow.hPow a b)
    -/
  · exact left_le_opow a hb
    /-
      🎉 no goals
    -/
    /-
      case a
      a b : Ordinal.{u_1}
      ha : LE.le Ordinal.omega0 a
      hb : LT.lt 0 b
      ⊢ LE.le b (HPow.hPow a b)
    -/
  · exact right_le_opow b (one_lt_omega0.trans_le ha)
    /-
      🎉 no goals
    -/


theorem card_opow_eq_of_omega0_le_right {a b : Ordinal} (ha : 1 < a) (hb : ω ≤ b) :
    (a ^ b).card = max a.card b.card := by
  /-
    a b : Ordinal.{u_1}
    ha : LT.lt 1 a
    hb : LE.le Ordinal.omega0 b
    ⊢ Eq (HPow.hPow a b).card (Max.max a.card b.card)
  -/
  apply (card_opow_le_of_omega0_le_right a hb).antisymm (max_le _ _) <;> apply card_le_card
    /-
      case a
      a b : Ordinal.{u_1}
      ha : LT.lt 1 a
      hb : LE.le Ordinal.omega0 b
      ⊢ LE.le a (HPow.hPow a b)
    -/
  · exact left_le_opow a (omega0_pos.trans_le hb)
    /-
      🎉 no goals
    -/
    /-
      case a
      a b : Ordinal.{u_1}
      ha : LT.lt 1 a
      hb : LE.le Ordinal.omega0 b
      ⊢ LE.le b (HPow.hPow a b)
    -/
  · exact right_le_opow b ha
    /-
      🎉 no goals
    -/


theorem card_omega0_opow {a : Ordinal} (h : a ≠ 0) : card (ω ^ a) = max ℵ₀ a.card := by
  /-
    a : Ordinal.{u_1}
    h : Ne a 0
    ⊢ Eq (HPow.hPow Ordinal.omega0 a).card (Max.max Cardinal.aleph0 a.card)
  -/
  rw [card_opow_eq_of_omega0_le_left le_rfl h.bot_lt, card_omega0]
  /-
    🎉 no goals
  -/


theorem card_opow_omega0 {a : Ordinal} (h : 1 < a) : card (a ^ ω) = max ℵ₀ a.card := by
  /-
    a : Ordinal.{u_1}
    h : LT.lt 1 a
    ⊢ Eq (HPow.hPow a Ordinal.omega0).card (Max.max Cardinal.aleph0 a.card)
  -/
  rw [card_opow_eq_of_omega0_le_right h le_rfl, card_omega0, max_comm]
  /-
    🎉 no goals
  -/


theorem principal_opow_omega (o : Ordinal) : Principal (· ^ ·) (ω_ o) := by
  /-
    o : Ordinal.{u_1}
    ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) (Ordinal.omega o)
  -/
  obtain rfl | ho := Ordinal.eq_zero_or_pos o
    /-
      case inl
      ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) (Ordinal.omega 0)
    -/
  · rw [omega_zero]
    /-
      case inl
      ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) Ordinal.omega0
    -/
    exact principal_opow_omega0
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) (Ordinal.omega o)
    -/
  · intro a b ha hb
    /-
      case inr
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      a b : Ordinal.{u_1}
      ha : LT.lt a (Ordinal.omega o)
      hb : LT.lt b (Ordinal.omega o)
      ⊢ LT.lt ((fun x1 x2 => HPow.hPow x1 x2) a b) (Ordinal.omega o)
    -/
    rw [lt_omega_iff_card_lt] at ha hb ⊢
    /-
      case inr
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      a b : Ordinal.{u_1}
      ha : LT.lt a.card (Cardinal.aleph o)
      hb : LT.lt b.card (Cardinal.aleph o)
      ⊢ LT.lt ((fun x1 x2 => HPow.hPow x1 x2) a b).card (Cardinal.aleph o)
    -/
    apply (card_opow_le a b).trans_lt (max_lt _ (max_lt ha hb))
    /-
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      a b : Ordinal.{u_1}
      ha : LT.lt a.card (Cardinal.aleph o)
      hb : LT.lt b.card (Cardinal.aleph o)
      ⊢ LT.lt Cardinal.aleph0 (Cardinal.aleph o)
    -/
    rwa [← aleph_zero, aleph_lt_aleph]
    /-
      🎉 no goals
    -/


theorem IsInitial.principal_opow {o : Ordinal} (h : IsInitial o) (ho : ω ≤ o) :
    Principal (· ^ ·) o := by
  /-
    o : Ordinal.{u_1}
    h : o.IsInitial
    ho : LE.le Ordinal.omega0 o
    ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) o
  -/
  obtain ⟨a, rfl⟩ := mem_range_omega_iff.2 ⟨ho, h⟩
  /-
    case intro
    a : Ordinal.{u_1}
    h : (Ordinal.omega a).IsInitial
    ho : LE.le Ordinal.omega0 (Ordinal.omega a)
    ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) (Ordinal.omega a)
  -/
  exact principal_opow_omega a
  /-
    🎉 no goals
  -/


theorem principal_opow_ord {c : Cardinal} (hc : ℵ₀ ≤ c) : Principal (· ^ ·) c.ord := by
  /-
    c : Cardinal.{u_1}
    hc : LE.le Cardinal.aleph0 c
    ⊢ Ordinal.Principal (fun x1 x2 => HPow.hPow x1 x2) c.ord
  -/
  apply (isInitial_ord c).principal_opow
  /-
    c : Cardinal.{u_1}
    hc : LE.le Cardinal.aleph0 c
    ⊢ LE.le Ordinal.omega0 c.ord
  -/
  rwa [omega0_le_ord]
  /-
    🎉 no goals
  -/


