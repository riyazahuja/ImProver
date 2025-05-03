/-- If `f` is strictly monotone both on `s` and `t`, with `s` to the left of `t` and the center
point belonging to both `s` and `t`, then `f` is strictly monotone on `s ∪ t` -/
protected theorem StrictMonoOn.union {s t : Set α} {c : α} (h₁ : StrictMonoOn f s)
    (h₂ : StrictMonoOn f t) (hs : IsGreatest s c) (ht : IsLeast t c) : StrictMonoOn f (s ∪ t) := by
  have A : ∀ x, x ∈ s ∪ t → x ≤ c → x ∈ s := by
    intro x hx hxc
    cases hx
    · assumption
    rcases eq_or_lt_of_le hxc with (rfl | h'x)
    · exact hs.1
    exact (lt_irrefl _ (h'x.trans_le (ht.2 (by assumption)))).elim
  have B : ∀ x, x ∈ s ∪ t → c ≤ x → x ∈ t := by
    intro x hx hxc
    match hx with
    | Or.inr hx => exact hx
    | Or.inl hx =>
      rcases eq_or_lt_of_le hxc with (rfl | h'x)
      · exact ht.1
      exact (lt_irrefl _ (h'x.trans_le (hs.2 hx))).elim
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    s t : Set α
    c : α
    h₁ : StrictMonoOn f s
    h₂ : StrictMonoOn f t
    hs : IsGreatest s c
    ht : IsLeast t c
    A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
    B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
    ⊢ StrictMonoOn f (Union.union s t)
  -/
  intro x hx y hy hxy
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    s t : Set α
    c : α
    h₁ : StrictMonoOn f s
    h₂ : StrictMonoOn f t
    hs : IsGreatest s c
    ht : IsLeast t c
    A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
    B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
    x : α
    hx : Membership.mem (Union.union s t) x
    y : α
    hy : Membership.mem (Union.union s t) y
    hxy : LT.lt x y
    ⊢ LT.lt (f x) (f y)
  -/
  rcases lt_or_le x c with (hxc | hcx)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : StrictMonoOn f s
      h₂ : StrictMonoOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LT.lt x y
      hxc : LT.lt x c
      ⊢ LT.lt (f x) (f y)
    -/
  · have xs : x ∈ s := A _ hx hxc.le
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : StrictMonoOn f s
      h₂ : StrictMonoOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LT.lt x y
      hxc : LT.lt x c
      xs : Membership.mem s x
      ⊢ LT.lt (f x) (f y)
    -/
    rcases lt_or_le y c with (hyc | hcy)
      /-
        case inl.inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : Preorder β
        f : α → β
        s t : Set α
        c : α
        h₁ : StrictMonoOn f s
        h₂ : StrictMonoOn f t
        hs : IsGreatest s c
        ht : IsLeast t c
        A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
        B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
        x : α
        hx : Membership.mem (Union.union s t) x
        y : α
        hy : Membership.mem (Union.union s t) y
        hxy : LT.lt x y
        hxc : LT.lt x c
        xs : Membership.mem s x
        hyc : LT.lt y c
        ⊢ LT.lt (f x) (f y)
      -/
    · exact h₁ xs (A _ hy hyc.le) hxy
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : Preorder β
        f : α → β
        s t : Set α
        c : α
        h₁ : StrictMonoOn f s
        h₂ : StrictMonoOn f t
        hs : IsGreatest s c
        ht : IsLeast t c
        A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
        B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
        x : α
        hx : Membership.mem (Union.union s t) x
        y : α
        hy : Membership.mem (Union.union s t) y
        hxy : LT.lt x y
        hxc : LT.lt x c
        xs : Membership.mem s x
        hcy : LE.le c y
        ⊢ LT.lt (f x) (f y)
      -/
    · exact (h₁ xs hs.1 hxc).trans_le (h₂.monotoneOn ht.1 (B _ hy hcy) hcy)
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : StrictMonoOn f s
      h₂ : StrictMonoOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LT.lt x y
      hcx : LE.le c x
      ⊢ LT.lt (f x) (f y)
    -/
  · have xt : x ∈ t := B _ hx hcx
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : StrictMonoOn f s
      h₂ : StrictMonoOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LT.lt x y
      hcx : LE.le c x
      xt : Membership.mem t x
      ⊢ LT.lt (f x) (f y)
    -/
    have yt : y ∈ t := B _ hy (hcx.trans hxy.le)
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : StrictMonoOn f s
      h₂ : StrictMonoOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LT.lt x y
      hcx : LE.le c x
      xt : Membership.mem t x
      yt : Membership.mem t y
      ⊢ LT.lt (f x) (f y)
    -/
    exact h₂ xt yt hxy
    /-
      🎉 no goals
    -/


/-- If `f` is strictly monotone both on `(-∞, a]` and `[a, ∞)`, then it is strictly monotone on the
whole line. -/
protected theorem StrictMonoOn.Iic_union_Ici (h₁ : StrictMonoOn f (Iic a))
    (h₂ : StrictMonoOn f (Ici a)) : StrictMono f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    a : α
    f : α → β
    h₁ : StrictMonoOn f (Set.Iic a)
    h₂ : StrictMonoOn f (Set.Ici a)
    ⊢ StrictMono f
  -/
  rw [← strictMonoOn_univ, ← @Iic_union_Ici _ _ a]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    a : α
    f : α → β
    h₁ : StrictMonoOn f (Set.Iic a)
    h₂ : StrictMonoOn f (Set.Ici a)
    ⊢ StrictMonoOn f (Union.union (Set.Iic a) (Set.Ici a))
  -/
  exact StrictMonoOn.union h₁ h₂ isGreatest_Iic isLeast_Ici
  /-
    🎉 no goals
  -/


/-- If `f` is strictly antitone both on `s` and `t`, with `s` to the left of `t` and the center
point belonging to both `s` and `t`, then `f` is strictly antitone on `s ∪ t` -/
protected theorem StrictAntiOn.union {s t : Set α} {c : α} (h₁ : StrictAntiOn f s)
    (h₂ : StrictAntiOn f t) (hs : IsGreatest s c) (ht : IsLeast t c) : StrictAntiOn f (s ∪ t) :=
  (h₁.dual_right.union h₂.dual_right hs ht).dual_right


/-- If `f` is strictly antitone both on `(-∞, a]` and `[a, ∞)`, then it is strictly antitone on the
whole line. -/
protected theorem StrictAntiOn.Iic_union_Ici (h₁ : StrictAntiOn f (Iic a))
    (h₂ : StrictAntiOn f (Ici a)) : StrictAnti f :=
  (h₁.dual_right.Iic_union_Ici h₂.dual_right).dual_right


/-- If `f` is monotone both on `s` and `t`, with `s` to the left of `t` and the center
point belonging to both `s` and `t`, then `f` is monotone on `s ∪ t` -/
protected theorem MonotoneOn.union_right {s t : Set α} {c : α} (h₁ : MonotoneOn f s)
    (h₂ : MonotoneOn f t) (hs : IsGreatest s c) (ht : IsLeast t c) : MonotoneOn f (s ∪ t) := by
  have A : ∀ x, x ∈ s ∪ t → x ≤ c → x ∈ s := by
    intro x hx hxc
    cases hx
    · assumption
    rcases eq_or_lt_of_le hxc with (rfl | h'x)
    · exact hs.1
    exact (lt_irrefl _ (h'x.trans_le (ht.2 (by assumption)))).elim
  have B : ∀ x, x ∈ s ∪ t → c ≤ x → x ∈ t := by
    intro x hx hxc
    match hx with
    | Or.inr hx => exact hx
    | Or.inl hx =>
      rcases eq_or_lt_of_le hxc with (rfl | h'x)
      · exact ht.1
      exact (lt_irrefl _ (h'x.trans_le (hs.2 hx))).elim
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    s t : Set α
    c : α
    h₁ : MonotoneOn f s
    h₂ : MonotoneOn f t
    hs : IsGreatest s c
    ht : IsLeast t c
    A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
    B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
    ⊢ MonotoneOn f (Union.union s t)
  -/
  intro x hx y hy hxy
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    f : α → β
    s t : Set α
    c : α
    h₁ : MonotoneOn f s
    h₂ : MonotoneOn f t
    hs : IsGreatest s c
    ht : IsLeast t c
    A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
    B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
    x : α
    hx : Membership.mem (Union.union s t) x
    y : α
    hy : Membership.mem (Union.union s t) y
    hxy : LE.le x y
    ⊢ LE.le (f x) (f y)
  -/
  rcases lt_or_le x c with (hxc | hcx)
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : MonotoneOn f s
      h₂ : MonotoneOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LE.le x y
      hxc : LT.lt x c
      ⊢ LE.le (f x) (f y)
    -/
  · have xs : x ∈ s := A _ hx hxc.le
    /-
      case inl
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : MonotoneOn f s
      h₂ : MonotoneOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LE.le x y
      hxc : LT.lt x c
      xs : Membership.mem s x
      ⊢ LE.le (f x) (f y)
    -/
    rcases lt_or_le y c with (hyc | hcy)
      /-
        case inl.inl
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : Preorder β
        f : α → β
        s t : Set α
        c : α
        h₁ : MonotoneOn f s
        h₂ : MonotoneOn f t
        hs : IsGreatest s c
        ht : IsLeast t c
        A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
        B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
        x : α
        hx : Membership.mem (Union.union s t) x
        y : α
        hy : Membership.mem (Union.union s t) y
        hxy : LE.le x y
        hxc : LT.lt x c
        xs : Membership.mem s x
        hyc : LT.lt y c
        ⊢ LE.le (f x) (f y)
      -/
    · exact h₁ xs (A _ hy hyc.le) hxy
      /-
        🎉 no goals
      -/
      /-
        case inl.inr
        α : Type u_1
        β : Type u_2
        inst✝¹ : LinearOrder α
        inst✝ : Preorder β
        f : α → β
        s t : Set α
        c : α
        h₁ : MonotoneOn f s
        h₂ : MonotoneOn f t
        hs : IsGreatest s c
        ht : IsLeast t c
        A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
        B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
        x : α
        hx : Membership.mem (Union.union s t) x
        y : α
        hy : Membership.mem (Union.union s t) y
        hxy : LE.le x y
        hxc : LT.lt x c
        xs : Membership.mem s x
        hcy : LE.le c y
        ⊢ LE.le (f x) (f y)
      -/
    · exact (h₁ xs hs.1 hxc.le).trans (h₂ ht.1 (B _ hy hcy) hcy)
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : MonotoneOn f s
      h₂ : MonotoneOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LE.le x y
      hcx : LE.le c x
      ⊢ LE.le (f x) (f y)
    -/
  · have xt : x ∈ t := B _ hx hcx
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : MonotoneOn f s
      h₂ : MonotoneOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LE.le x y
      hcx : LE.le c x
      xt : Membership.mem t x
      ⊢ LE.le (f x) (f y)
    -/
    have yt : y ∈ t := B _ hy (hcx.trans hxy)
    /-
      case inr
      α : Type u_1
      β : Type u_2
      inst✝¹ : LinearOrder α
      inst✝ : Preorder β
      f : α → β
      s t : Set α
      c : α
      h₁ : MonotoneOn f s
      h₂ : MonotoneOn f t
      hs : IsGreatest s c
      ht : IsLeast t c
      A : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le x c → Membership.mem …
      B : ∀ (x : α), Membership.mem (Union.union s t) x → LE.le c x → Membership.mem …
      x : α
      hx : Membership.mem (Union.union s t) x
      y : α
      hy : Membership.mem (Union.union s t) y
      hxy : LE.le x y
      hcx : LE.le c x
      xt : Membership.mem t x
      yt : Membership.mem t y
      ⊢ LE.le (f x) (f y)
    -/
    exact h₂ xt yt hxy
    /-
      🎉 no goals
    -/


/-- If `f` is monotone both on `(-∞, a]` and `[a, ∞)`, then it is monotone on the whole line. -/
protected theorem MonotoneOn.Iic_union_Ici (h₁ : MonotoneOn f (Iic a)) (h₂ : MonotoneOn f (Ici a)) :
    Monotone f := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    a : α
    f : α → β
    h₁ : MonotoneOn f (Set.Iic a)
    h₂ : MonotoneOn f (Set.Ici a)
    ⊢ Monotone f
  -/
  rw [← monotoneOn_univ, ← @Iic_union_Ici _ _ a]
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : LinearOrder α
    inst✝ : Preorder β
    a : α
    f : α → β
    h₁ : MonotoneOn f (Set.Iic a)
    h₂ : MonotoneOn f (Set.Ici a)
    ⊢ MonotoneOn f (Union.union (Set.Iic a) (Set.Ici a))
  -/
  exact MonotoneOn.union_right h₁ h₂ isGreatest_Iic isLeast_Ici
  /-
    🎉 no goals
  -/


/-- If `f` is antitone both on `s` and `t`, with `s` to the left of `t` and the center
point belonging to both `s` and `t`, then `f` is antitone on `s ∪ t` -/
protected theorem AntitoneOn.union_right {s t : Set α} {c : α} (h₁ : AntitoneOn f s)
    (h₂ : AntitoneOn f t) (hs : IsGreatest s c) (ht : IsLeast t c) : AntitoneOn f (s ∪ t) :=
  (h₁.dual_right.union_right h₂.dual_right hs ht).dual_right


/-- If `f` is antitone both on `(-∞, a]` and `[a, ∞)`, then it is antitone on the whole line. -/
protected theorem AntitoneOn.Iic_union_Ici (h₁ : AntitoneOn f (Iic a)) (h₂ : AntitoneOn f (Ici a)) :
    Antitone f :=
  (h₁.dual_right.Iic_union_Ici h₂.dual_right).dual_right

