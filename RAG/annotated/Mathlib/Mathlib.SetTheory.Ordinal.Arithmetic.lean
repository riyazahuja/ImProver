@[simp]
theorem lift_add (a b : Ordinal.{v}) : lift.{u} (a + b) = lift.{u} a + lift.{u} b :=
  Quotient.inductionOn₂ a b fun ⟨_α, _r, _⟩ ⟨_β, _s, _⟩ =>
    Quotient.sound
      ⟨(RelIso.preimage Equiv.ulift _).trans
          (RelIso.sumLexCongr (RelIso.preimage Equiv.ulift _) (RelIso.preimage Equiv.ulift _)).symm⟩


@[simp]
theorem lift_succ (a : Ordinal.{v}) : lift.{u} (succ a) = succ (lift.{u} a) := by
  /-
    a : Ordinal.{v}
    ⊢ Eq (Ordinal.lift.{u, v} (Order.succ a)) (Order.succ (Ordinal.lift.{u, v} a))
  -/
  rw [← add_one_eq_succ, lift_add, lift_one]
  /-
    a : Ordinal.{v}
    ⊢ Eq (HAdd.hAdd (Ordinal.lift.{u, v} a) 1) (Order.succ (Ordinal.lift.{u, v} a))
  -/
  rfl
  /-
    🎉 no goals
  -/


instance instAddLeftReflectLE :
    AddLeftReflectLE Ordinal.{u} where
  elim c a b := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      r : α → α → Prop
      s : β → β → Prop
      t : γ → γ → Prop
      c a b : Ordinal.{u}
      ⊢ LE.le (HAdd.hAdd c a) (HAdd.hAdd c b) → LE.le a b
    -/
    refine inductionOn₃ a b c fun α r _ β s _ γ t _ ⟨f⟩ ↦ ?_
    have H₁ a : f (Sum.inl a) = Sum.inl a := by
      simpa using ((InitialSeg.leAdd t r).trans f).eq (InitialSeg.leAdd t s) a
    have H₂ a : ∃ b, f (Sum.inr a) = Sum.inr b := by
      generalize hx : f (Sum.inr a) = x
      obtain x | x := x
      · rw [← H₁, f.inj] at hx
        contradiction
      · exact ⟨x, rfl⟩
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      r✝ : α✝ → α✝ → Prop
      s✝ : β✝ → β✝ → Prop
      t✝ : γ✝ → γ✝ → Prop
      c a b : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝³ : IsWellOrder α r
      β : Type u
      s : β → β → Prop
      x✝² : IsWellOrder β s
      γ : Type u
      t : γ → γ → Prop
      x✝¹ : IsWellOrder γ t
      x✝ : LE.le (HAdd.hAdd (Ordinal.type t) (Ordinal.type r)) (HAdd.hAdd (Ordinal.t …
      f : InitialSeg (Sum.Lex t r) (Sum.Lex t s)
      H₁ : ∀ (a : γ), Eq (f (Sum.inl a)) (Sum.inl a)
      H₂ : ∀ (a : α), Exists fun b => Eq (f (Sum.inr a)) (Sum.inr b)
      ⊢ LE.le (Ordinal.type r) (Ordinal.type s)
    -/
    choose g hg using H₂
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      r✝ : α✝ → α✝ → Prop
      s✝ : β✝ → β✝ → Prop
      t✝ : γ✝ → γ✝ → Prop
      c a b : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝³ : IsWellOrder α r
      β : Type u
      s : β → β → Prop
      x✝² : IsWellOrder β s
      γ : Type u
      t : γ → γ → Prop
      x✝¹ : IsWellOrder γ t
      x✝ : LE.le (HAdd.hAdd (Ordinal.type t) (Ordinal.type r)) (HAdd.hAdd (Ordinal.t …
      f : InitialSeg (Sum.Lex t r) (Sum.Lex t s)
      H₁ : ∀ (a : γ), Eq (f (Sum.inl a)) (Sum.inl a)
      g : α → β
      hg : ∀ (a : α), Eq (f (Sum.inr a)) (Sum.inr (g a))
      ⊢ LE.le (Ordinal.type r) (Ordinal.type s)
    -/
    refine (RelEmbedding.ofMonotone g fun _ _ h ↦ ?_).ordinal_type_le
    /-
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      r✝ : α✝ → α✝ → Prop
      s✝ : β✝ → β✝ → Prop
      t✝ : γ✝ → γ✝ → Prop
      c a b : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝⁵ : IsWellOrder α r
      β : Type u
      s : β → β → Prop
      x✝⁴ : IsWellOrder β s
      γ : Type u
      t : γ → γ → Prop
      x✝³ : IsWellOrder γ t
      x✝² : LE.le (HAdd.hAdd (Ordinal.type t) (Ordinal.type r)) (HAdd.hAdd (Ordinal. …
      f : InitialSeg (Sum.Lex t r) (Sum.Lex t s)
      H₁ : ∀ (a : γ), Eq (f (Sum.inl a)) (Sum.inl a)
      g : α → β
      hg : ∀ (a : α), Eq (f (Sum.inr a)) (Sum.inr (g a))
      x✝¹ x✝ : α
      h : r x✝¹ x✝
      ⊢ s (g x✝¹) (g x✝)
    -/
    rwa [← @Sum.lex_inr_inr _ t _ s, ← hg, ← hg, f.map_rel_iff, Sum.lex_inr_inr]
    /-
      🎉 no goals
    -/


instance : IsLeftCancelAdd Ordinal where
                                /-
                                  α : Type u_1
                                  β : Type u_2
                                  γ : Type u_3
                                  r : α → α → Prop
                                  s : β → β → Prop
                                  t : γ → γ → Prop
                                  a b c : Ordinal.{u_4}
                                  h : Eq (HAdd.hAdd a b) (HAdd.hAdd a c)
                                  ⊢ Eq b c
                                -/
  add_left_cancel a b c h := by simpa only [le_antisymm_iff, add_le_add_iff_left] using h
                                /-
                                  🎉 no goals
                                -/


@[deprecated add_left_cancel_iff (since := "2024-12-11")]
protected theorem add_left_cancel (a) {b c : Ordinal} : a + b = a + c ↔ b = c :=
  add_left_cancel_iff


private theorem add_lt_add_iff_left' (a) {b c : Ordinal} : a + b < a + c ↔ b < c := by
  /-
    a b c : Ordinal.{u_4}
    ⊢ Iff (LT.lt (HAdd.hAdd a b) (HAdd.hAdd a c)) (LT.lt b c)
  -/
  rw [← not_le, ← not_le, add_le_add_iff_left]
  /-
    🎉 no goals
  -/


instance instAddLeftStrictMono : AddLeftStrictMono Ordinal.{u} :=
  ⟨fun a _b _c ↦ (add_lt_add_iff_left' a).2⟩


instance instAddLeftReflectLT : AddLeftReflectLT Ordinal.{u} :=
  ⟨fun a _b _c ↦ (add_lt_add_iff_left' a).1⟩


instance instAddRightReflectLT : AddRightReflectLT Ordinal.{u} :=
  ⟨fun _a _b _c ↦ lt_imp_lt_of_le_imp_le fun h => add_le_add_right h _⟩


theorem add_le_add_iff_right {a b : Ordinal} : ∀ n : ℕ, a + n ≤ b + n ↔ a ≤ b
            /-
              a b : Ordinal.{u_4}
              ⊢ Iff (LE.le (HAdd.hAdd a ↑0) (HAdd.hAdd b ↑0)) (LE.le a b)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | n + 1 => by
    /-
      a b : Ordinal.{u_4}
      n : Nat
      ⊢ Iff (LE.le (HAdd.hAdd a ↑(HAdd.hAdd n 1)) (HAdd.hAdd b ↑(HAdd.hAdd n 1))) (L …
    -/
    simp only [natCast_succ, add_succ, add_succ, succ_le_succ_iff, add_le_add_iff_right]
    /-
      🎉 no goals
    -/


theorem add_right_cancel {a b : Ordinal} (n : ℕ) : a + n = b + n ↔ a = b := by
  /-
    a b : Ordinal.{u_4}
    n : Nat
    ⊢ Iff (Eq (HAdd.hAdd a ↑n) (HAdd.hAdd b ↑n)) (Eq a b)
  -/
  simp only [le_antisymm_iff, add_le_add_iff_right]
  /-
    🎉 no goals
  -/


theorem add_eq_zero_iff {a b : Ordinal} : a + b = 0 ↔ a = 0 ∧ b = 0 :=
  inductionOn₂ a b fun α r _ β s _ => by
    /-
      a b : Ordinal.{u_4}
      α : Type u_4
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type u_4
      s : β → β → Prop
      x✝ : IsWellOrder β s
      ⊢ Iff (Eq (HAdd.hAdd (Ordinal.type r) (Ordinal.type s)) 0) (And (Eq (Ordinal.t …
    -/
    simp_rw [← type_sum_lex, type_eq_zero_iff_isEmpty]
    /-
      a b : Ordinal.{u_4}
      α : Type u_4
      r : α → α → Prop
      x✝¹ : IsWellOrder α r
      β : Type u_4
      s : β → β → Prop
      x✝ : IsWellOrder β s
      ⊢ Iff (IsEmpty (Sum α β)) (And (IsEmpty α) (IsEmpty β))
    -/
    exact isEmpty_sum
    /-
      🎉 no goals
    -/


theorem left_eq_zero_of_add_eq_zero {a b : Ordinal} (h : a + b = 0) : a = 0 :=
  (add_eq_zero_iff.1 h).1


theorem right_eq_zero_of_add_eq_zero {a b : Ordinal} (h : a + b = 0) : b = 0 :=
  (add_eq_zero_iff.1 h).2


open Classical in
/-- The ordinal predecessor of `o` is `o'` if `o = succ o'`,
  and `o` otherwise. -/
def pred (o : Ordinal) : Ordinal :=
  if h : ∃ a, o = succ a then Classical.choose h else o


@[simp]
theorem pred_succ (o) : pred (succ o) = o := by
  /-
    o : Ordinal.{u_4}
    ⊢ Eq (Order.succ o).pred o
  -/
  have h : ∃ a, succ o = succ a := ⟨_, rfl⟩
  /-
    o : Ordinal.{u_4}
    h : Exists fun a => Eq (Order.succ o) (Order.succ a)
    ⊢ Eq (Order.succ o).pred o
  -/
  simpa only [pred, dif_pos h] using (succ_injective <| Classical.choose_spec h).symm
  /-
    🎉 no goals
  -/


theorem pred_le_self (o) : pred o ≤ o := by
  classical
  exact if h : ∃ a, o = succ a then by
    let ⟨a, e⟩ := h
    rw [e, pred_succ]; exact le_succ a
  else by rw [pred, dif_neg h]


theorem pred_eq_iff_not_succ {o} : pred o = o ↔ ¬∃ a, o = succ a :=
                       /-
                         o : Ordinal.{u_4}
                         e : Eq o.pred o
                         x✝ : Exists fun a => Eq o (Order.succ a)
                         a : Ordinal.{u_4}
                         e' : Eq o (Order.succ a)
                         ⊢ False
                       -/
  ⟨fun e ⟨a, e'⟩ => by rw [e', pred_succ] at e; exact (lt_succ a).ne e, fun h => dif_neg h⟩
                                                /-
                                                  🎉 no goals
                                                -/


theorem pred_eq_iff_not_succ' {o} : pred o = o ↔ ∀ a, o ≠ succ a := by
  /-
    o : Ordinal.{u_4}
    ⊢ Iff (Eq o.pred o) (∀ (a : Ordinal.{u_4}), Ne o (Order.succ a))
  -/
  simpa using pred_eq_iff_not_succ
  /-
    🎉 no goals
  -/


theorem pred_lt_iff_is_succ {o} : pred o < o ↔ ∃ a, o = succ a :=
                /-
                  o : Ordinal.{u_4}
                  ⊢ Iff (LT.lt o.pred o) (Not (Eq o.pred o))
                -/
  Iff.trans (by simp only [le_antisymm_iff, pred_le_self, true_and, not_le])
                /-
                  🎉 no goals
                -/
    (iff_not_comm.1 pred_eq_iff_not_succ).symm


@[simp]
theorem pred_zero : pred 0 = 0 :=
  pred_eq_iff_not_succ'.2 fun a => (succ_ne_zero a).symm


theorem succ_pred_iff_is_succ {o} : succ (pred o) = o ↔ ∃ a, o = succ a :=
                                          /-
                                            o : Ordinal.{u_4}
                                            x✝ : Exists fun a => Eq o (Order.succ a)
                                            a : Ordinal.{u_4}
                                            e : Eq o (Order.succ a)
                                            ⊢ Eq (Order.succ o.pred) o
                                          -/
  ⟨fun e => ⟨_, e.symm⟩, fun ⟨a, e⟩ => by simp only [e, pred_succ]⟩
                                          /-
                                            🎉 no goals
                                          -/


theorem succ_lt_of_not_succ {o b : Ordinal} (h : ¬∃ a, o = succ a) : succ b < o ↔ b < o :=
  ⟨(lt_succ b).trans, fun l => lt_of_le_of_ne (succ_le_of_lt l) fun e => h ⟨_, e.symm⟩⟩


theorem lt_pred {a b} : a < pred b ↔ succ a < b := by
  classical
  exact if h : ∃ a, b = succ a then by
    let ⟨c, e⟩ := h
    rw [e, pred_succ, succ_lt_succ_iff]
  else by simp only [pred, dif_neg h, succ_lt_of_not_succ h]


theorem pred_le {a b} : pred a ≤ b ↔ a ≤ succ b :=
  le_iff_le_iff_lt_iff_lt.2 lt_pred


@[simp]
theorem lift_is_succ {o : Ordinal.{v}} : (∃ a, lift.{u} o = succ a) ↔ ∃ a, o = succ a :=
  ⟨fun ⟨a, h⟩ =>
    let ⟨b, e⟩ := mem_range_lift_of_le <| show a ≤ lift.{u} o from le_of_lt <| h.symm ▸ lt_succ a
                                 /-
                                   o : Ordinal.{v}
                                   x✝ : Exists fun a => Eq (Ordinal.lift.{u, v} o) (Order.succ a)
                                   a : Ordinal.{max v u}
                                   h : Eq (Ordinal.lift.{u, v} o) (Order.succ a)
                                   b : Ordinal.{v}
                                   e : Eq (Ordinal.lift.{u, v} b) a
                                   ⊢ Eq (Ordinal.lift.{u, v} o) (Ordinal.lift.{u, v} (Order.succ b))
                                 -/
    ⟨b, (lift_inj.{u,v}).1 <| by rw [h, ← e, lift_succ]⟩,
                                 /-
                                   🎉 no goals
                                 -/
                                  /-
                                    o : Ordinal.{v}
                                    x✝ : Exists fun a => Eq o (Order.succ a)
                                    a : Ordinal.{v}
                                    h : Eq o (Order.succ a)
                                    ⊢ Eq (Ordinal.lift.{u, v} o) (Order.succ (Ordinal.lift.{u, v} a))
                                  -/
    fun ⟨a, h⟩ => ⟨lift.{u} a, by simp only [h, lift_succ]⟩⟩
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem lift_pred (o : Ordinal.{v}) : lift.{u} (pred o) = pred (lift.{u} o) := by
  classical
  exact if h : ∃ a, o = succ a then by cases' h with a e; simp only [e, pred_succ, lift_succ]
  else by rw [pred_eq_iff_not_succ.2 h, pred_eq_iff_not_succ.2 (mt lift_is_succ.1 h)]


/-- A limit ordinal is an ordinal which is not zero and not a successor.

TODO: deprecate this in favor of `Order.IsSuccLimit`. -/
def IsLimit (o : Ordinal) : Prop :=
  IsSuccLimit o


theorem isLimit_iff {o} : IsLimit o ↔ o ≠ 0 ∧ IsSuccPrelimit o := by
  /-
    o : Ordinal.{u_4}
    ⊢ Iff o.IsLimit (And (Ne o 0) (Order.IsSuccPrelimit o))
  -/
  simp [IsLimit, IsSuccLimit]
  /-
    🎉 no goals
  -/


theorem IsLimit.isSuccPrelimit {o} (h : IsLimit o) : IsSuccPrelimit o :=
  IsSuccLimit.isSuccPrelimit h


@[deprecated IsLimit.isSuccPrelimit (since := "2024-09-05")]
alias IsLimit.isSuccLimit := IsLimit.isSuccPrelimit


theorem IsLimit.succ_lt {o a : Ordinal} (h : IsLimit o) : a < o → succ a < o :=
  IsSuccLimit.succ_lt h


theorem isSuccPrelimit_zero : IsSuccPrelimit (0 : Ordinal) := isSuccPrelimit_bot


@[deprecated isSuccPrelimit_zero (since := "2024-09-05")]
alias isSuccLimit_zero := isSuccPrelimit_zero


theorem not_zero_isLimit : ¬IsLimit 0 :=
  not_isSuccLimit_bot


theorem not_succ_isLimit (o) : ¬IsLimit (succ o) :=
  not_isSuccLimit_succ o


theorem not_succ_of_isLimit {o} (h : IsLimit o) : ¬∃ a, o = succ a
  | ⟨a, e⟩ => not_succ_isLimit a (e ▸ h)


theorem succ_lt_of_isLimit {o a : Ordinal} (h : IsLimit o) : succ a < o ↔ a < o :=
  IsSuccLimit.succ_lt_iff h


theorem le_succ_of_isLimit {o} (h : IsLimit o) {a} : o ≤ succ a ↔ o ≤ a :=
  le_iff_le_iff_lt_iff_lt.2 <| succ_lt_of_isLimit h


theorem limit_le {o} (h : IsLimit o) {a} : o ≤ a ↔ ∀ x < o, x ≤ a :=
  ⟨fun h _x l => l.le.trans h, fun H =>
    (le_succ_of_isLimit h).1 <| le_of_not_lt fun hn => not_lt_of_le (H _ hn) (lt_succ a)⟩


theorem lt_limit {o} (h : IsLimit o) {a} : a < o ↔ ∃ x < o, a < x := by
  -- Porting note: `bex_def` is required.
  /-
    o : Ordinal.{u_4}
    h : o.IsLimit
    a : Ordinal.{u_4}
    ⊢ Iff (LT.lt a o) (Exists fun x => And (LT.lt x o) (LT.lt a x))
  -/
  simpa only [not_forall₂, not_le, bex_def] using not_congr (@limit_le _ h a)
  /-
    🎉 no goals
  -/


@[simp]
theorem lift_isLimit (o : Ordinal.{v}) : IsLimit (lift.{u,v} o) ↔ IsLimit o :=
  liftInitialSeg.isSuccLimit_apply_iff


theorem IsLimit.pos {o : Ordinal} (h : IsLimit o) : 0 < o :=
  IsSuccLimit.bot_lt h


theorem IsLimit.ne_zero {o : Ordinal} (h : IsLimit o) : o ≠ 0 :=
  h.pos.ne'


theorem IsLimit.one_lt {o : Ordinal} (h : IsLimit o) : 1 < o := by
  /-
    o : Ordinal.{u_4}
    h : o.IsLimit
    ⊢ LT.lt 1 o
  -/
  simpa only [succ_zero] using h.succ_lt h.pos
  /-
    🎉 no goals
  -/


theorem IsLimit.nat_lt {o : Ordinal} (h : IsLimit o) : ∀ n : ℕ, (n : Ordinal) < o
  | 0 => h.pos
  | n + 1 => h.succ_lt (IsLimit.nat_lt h n)


theorem zero_or_succ_or_limit (o : Ordinal) : o = 0 ∨ (∃ a, o = succ a) ∨ IsLimit o := by
  /-
    o : Ordinal.{u_4}
    ⊢ Or (Eq o 0) (Or (Exists fun a => Eq o (Order.succ a)) o.IsLimit)
  -/
  simpa [eq_comm] using isMin_or_mem_range_succ_or_isSuccLimit o
  /-
    🎉 no goals
  -/


theorem isLimit_of_not_succ_of_ne_zero {o : Ordinal} (h : ¬∃ a, o = succ a) (h' : o ≠ 0) :
    IsLimit o := ((zero_or_succ_or_limit o).resolve_left h').resolve_left h

-- TODO: this is an iff with `IsSuccPrelimit`

theorem IsLimit.sSup_Iio {o : Ordinal} (h : IsLimit o) : sSup (Iio o) = o := by
  /-
    o : Ordinal.{u_4}
    h : o.IsLimit
    ⊢ Eq (SupSet.sSup (Set.Iio o)) o
  -/
  apply (csSup_le' (fun a ha ↦ le_of_lt ha)).antisymm
  /-
    o : Ordinal.{u_4}
    h : o.IsLimit
    ⊢ LE.le o (SupSet.sSup fun a => Preorder.toLT.1 a o)
  -/
  apply le_of_forall_lt
  /-
    case H
    o : Ordinal.{u_4}
    h : o.IsLimit
    ⊢ ∀ (c : Ordinal.{u_4}), LT.lt c o → LT.lt c (SupSet.sSup fun a => Preorder.to …
  -/
  intro a ha
  /-
    case H
    o : Ordinal.{u_4}
    h : o.IsLimit
    a : Ordinal.{u_4}
    ha : LT.lt a o
    ⊢ LT.lt a (SupSet.sSup fun a => Preorder.toLT.1 a o)
  -/
  exact (lt_succ a).trans_le (le_csSup bddAbove_Iio (h.succ_lt ha))
  /-
    🎉 no goals
  -/


theorem IsLimit.iSup_Iio {o : Ordinal} (h : IsLimit o) : ⨆ a : Iio o, a.1 = o := by
  /-
    o : Ordinal.{u_4}
    h : o.IsLimit
    ⊢ Eq (iSup fun a => ↑a) o
  -/
  rw [← sSup_eq_iSup', h.sSup_Iio]
  /-
    🎉 no goals
  -/


/-- Main induction principle of ordinals: if one can prove a property by
  induction at successor ordinals and at limit ordinals, then it holds for all ordinals. -/
@[elab_as_elim]
def limitRecOn {C : Ordinal → Sort*} (o : Ordinal) (H₁ : C 0) (H₂ : ∀ o, C o → C (succ o))
    (H₃ : ∀ o, IsLimit o → (∀ o' < o, C o') → C o) : C o := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    C : Ordinal.{?u.23959} → Sort u_4
    o : Ordinal.{?u.23959}
    H₁ : C 0
    H₂ : (o : Ordinal.{?u.23959}) → C o → C (Order.succ o)
    H₃ : (o : Ordinal.{?u.23959}) → o.IsLimit → ((o' : Ordinal.{?u.23959}) → LT.lt …
    ⊢ C o
  -/
  refine SuccOrder.limitRecOn o (fun a ha ↦ ?_) (fun a _ ↦ H₂ a) H₃
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    C : Ordinal.{?u.23959} → Sort u_4
    o : Ordinal.{?u.23959}
    H₁ : C 0
    H₂ : (o : Ordinal.{?u.23959}) → C o → C (Order.succ o)
    H₃ : (o : Ordinal.{?u.23959}) → o.IsLimit → ((o' : Ordinal.{?u.23959}) → LT.lt …
    a : Ordinal.{?u.23959}
    ha : IsMin a
    ⊢ C a
  -/
  convert H₁
  /-
    case h.e'_1
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    r : α → α → Prop
    s : β → β → Prop
    t : γ → γ → Prop
    C : Ordinal.{?u.23959} → Sort u_4
    o : Ordinal.{?u.23959}
    H₁ : C 0
    H₂ : (o : Ordinal.{?u.23959}) → C o → C (Order.succ o)
    H₃ : (o : Ordinal.{?u.23959}) → o.IsLimit → ((o' : Ordinal.{?u.23959}) → LT.lt …
    a : Ordinal.{?u.23959}
    ha : IsMin a
    ⊢ Eq a 0
  -/
  simpa using ha
  /-
    🎉 no goals
  -/


@[simp]
theorem limitRecOn_zero {C} (H₁ H₂ H₃) : @limitRecOn C 0 H₁ H₂ H₃ = H₁ :=
  SuccOrder.limitRecOn_isMin _ _ _ isMin_bot


@[simp]
theorem limitRecOn_succ {C} (o H₁ H₂ H₃) :
    @limitRecOn C (succ o) H₁ H₂ H₃ = H₂ o (@limitRecOn C o H₁ H₂ H₃) :=
  SuccOrder.limitRecOn_succ ..


@[simp]
theorem limitRecOn_limit {C} (o H₁ H₂ H₃ h) :
    @limitRecOn C o H₁ H₂ H₃ = H₃ o h fun x _h => @limitRecOn C x H₁ H₂ H₃ :=
  SuccOrder.limitRecOn_of_isSuccLimit ..


/-- Bounded recursion on ordinals. Similar to `limitRecOn`, with the assumption `o < l`
  added to all cases. The final term's domain is the ordinals below `l`. -/
@[elab_as_elim]
def boundedLimitRecOn {l : Ordinal} (lLim : l.IsLimit) {C : Iio l → Sort*} (o : Iio l)
    (H₁ : C ⟨0, lLim.pos⟩) (H₂ : (o : Iio l) → C o → C ⟨succ o, lLim.succ_lt o.2⟩)
    (H₃ : (o : Iio l) → IsLimit o → (Π o' < o, C o') → C o) : C o :=
  limitRecOn (C := fun p ↦ (h : p < l) → C ⟨p, h⟩) o.1 (fun _ ↦ H₁)
    (fun o ih h ↦ H₂ ⟨o, _⟩ <| ih <| (lt_succ o).trans h)
    (fun _o ho ih _ ↦ H₃ _ ho fun _o' h ↦ ih _ h _) o.2


@[simp]
theorem boundedLimitRec_zero {l} (lLim : l.IsLimit) {C} (H₁ H₂ H₃) :
    @boundedLimitRecOn l lLim C ⟨0, lLim.pos⟩ H₁ H₂ H₃ = H₁ := by
  /-
    l : Ordinal.{u_4}
    lLim : l.IsLimit
    C : ↑(Set.Iio l) → Sort u_5
    H₁ : C ⟨0, ⋯⟩
    H₂ : (o : ↑(Set.Iio l)) → C o → C ⟨Order.succ ↑o, ⋯⟩
    H₃ : (o : ↑(Set.Iio l)) → (↑o).IsLimit → ((o' : ↑(Set.Iio l)) → LT.lt o' o → C …
    ⊢ Eq (Ordinal.boundedLimitRecOn lLim ⟨0, ⋯⟩ H₁ H₂ H₃) H₁
  -/
  rw [boundedLimitRecOn, limitRecOn_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem boundedLimitRec_succ {l} (lLim : l.IsLimit) {C} (o H₁ H₂ H₃) :
    @boundedLimitRecOn l lLim C ⟨succ o.1, lLim.succ_lt o.2⟩ H₁ H₂ H₃ = H₂ o
    (@boundedLimitRecOn l lLim C o H₁ H₂ H₃) := by
  /-
    l : Ordinal.{u_4}
    lLim : l.IsLimit
    C : ↑(Set.Iio l) → Sort u_5
    o : ↑(Set.Iio l)
    H₁ : C ⟨0, ⋯⟩
    H₂ : (o : ↑(Set.Iio l)) → C o → C ⟨Order.succ ↑o, ⋯⟩
    H₃ : (o : ↑(Set.Iio l)) → (↑o).IsLimit → ((o' : ↑(Set.Iio l)) → LT.lt o' o → C …
    ⊢ Eq (Ordinal.boundedLimitRecOn lLim ⟨Order.succ ↑o, ⋯⟩ H₁ H₂ H₃) (H₂ o (Ordin …
  -/
  rw [boundedLimitRecOn, limitRecOn_succ]
  /-
    l : Ordinal.{u_4}
    lLim : l.IsLimit
    C : ↑(Set.Iio l) → Sort u_5
    o : ↑(Set.Iio l)
    H₁ : C ⟨0, ⋯⟩
    H₂ : (o : ↑(Set.Iio l)) → C o → C ⟨Order.succ ↑o, ⋯⟩
    H₃ : (o : ↑(Set.Iio l)) → (↑o).IsLimit → ((o' : ↑(Set.Iio l)) → LT.lt o' o → C …
    ⊢ Eq (H₂ ⟨↑o, ⋯⟩ ((↑o).limitRecOn (fun x => H₁) (fun o ih h => H₂ ⟨o, ⋯⟩ (ih ⋯ …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem boundedLimitRec_limit {l} (lLim : l.IsLimit) {C} (o H₁ H₂ H₃ oLim) :
    @boundedLimitRecOn l lLim C o H₁ H₂ H₃ = H₃ o oLim (fun x _ ↦
    @boundedLimitRecOn l lLim C x H₁ H₂ H₃) := by
  /-
    l : Ordinal.{u_4}
    lLim : l.IsLimit
    C : ↑(Set.Iio l) → Sort u_5
    o : ↑(Set.Iio l)
    H₁ : C ⟨0, ⋯⟩
    H₂ : (o : ↑(Set.Iio l)) → C o → C ⟨Order.succ ↑o, ⋯⟩
    H₃ : (o : ↑(Set.Iio l)) → (↑o).IsLimit → ((o' : ↑(Set.Iio l)) → LT.lt o' o → C …
    oLim : (↑o).IsLimit
    ⊢ Eq (Ordinal.boundedLimitRecOn lLim o H₁ H₂ H₃) (H₃ o oLim fun x x_1 => Ordin …
  -/
  rw [boundedLimitRecOn, limitRecOn_limit]
  /-
    l : Ordinal.{u_4}
    lLim : l.IsLimit
    C : ↑(Set.Iio l) → Sort u_5
    o : ↑(Set.Iio l)
    H₁ : C ⟨0, ⋯⟩
    H₂ : (o : ↑(Set.Iio l)) → C o → C ⟨Order.succ ↑o, ⋯⟩
    H₃ : (o : ↑(Set.Iio l)) → (↑o).IsLimit → ((o' : ↑(Set.Iio l)) → LT.lt o' o → C …
    oLim : (↑o).IsLimit
    ⊢ Eq (H₃ ⟨↑o, ⋯⟩ ?h fun _o' h => (fun x _h => x.limitRecOn (fun x => H₁) (fun  …
  -/
  rfl
  /-
    🎉 no goals
  -/


instance orderTopToTypeSucc (o : Ordinal) : OrderTop (succ o).toType :=
  @OrderTop.mk _ _ (Top.mk _) le_enum_succ


theorem enum_succ_eq_top {o : Ordinal} :
    enum (α := (succ o).toType) (· < ·) ⟨o, type_toType _ ▸ lt_succ o⟩ = ⊤ :=
  rfl


theorem has_succ_of_type_succ_lt {α} {r : α → α → Prop} [wo : IsWellOrder α r]
    (h : ∀ a < type r, succ a < type r) (x : α) : ∃ y, r x y := by
  /-
    α : Type u_4
    r : α → α → Prop
    wo : IsWellOrder α r
    h : ∀ (a : Ordinal.{u_4}), LT.lt a (Ordinal.type r) → LT.lt (Order.succ a) (Or …
    x : α
    ⊢ Exists fun y => r x y
  -/
  use enum r ⟨succ (typein r x), h _ (typein_lt_type r x)⟩
  /-
    case h
    α : Type u_4
    r : α → α → Prop
    wo : IsWellOrder α r
    h : ∀ (a : Ordinal.{u_4}), LT.lt a (Ordinal.type r) → LT.lt (Order.succ a) (Or …
    x : α
    ⊢ r x ((Ordinal.enum r) ⟨Order.succ ((Ordinal.typein r).toRelEmbedding x), ⋯⟩)
  -/
  convert enum_lt_enum (o₁ := ⟨_, typein_lt_type r x⟩) (o₂ := ⟨_, h _ (typein_lt_type r x)⟩).mpr _
    /-
      case h.e'_1
      α : Type u_4
      r : α → α → Prop
      wo : IsWellOrder α r
      h : ∀ (a : Ordinal.{u_4}), LT.lt a (Ordinal.type r) → LT.lt (Order.succ a) (Or …
      x : α
      ⊢ Eq x ((Ordinal.enum r) ⟨(Ordinal.typein r).toRelEmbedding x, ⋯⟩)
    -/
  · rw [enum_typein]
    /-
      🎉 no goals
    -/
    /-
      case h
      α : Type u_4
      r : α → α → Prop
      wo : IsWellOrder α r
      h : ∀ (a : Ordinal.{u_4}), LT.lt a (Ordinal.type r) → LT.lt (Order.succ a) (Or …
      x : α
      ⊢ LT.lt ⟨(Ordinal.typein r).toRelEmbedding x, ⋯⟩ ⟨Order.succ ((Ordinal.typein  …
    -/
  · rw [Subtype.mk_lt_mk, lt_succ_iff]
    /-
      🎉 no goals
    -/


theorem toType_noMax_of_succ_lt {o : Ordinal} (ho : ∀ a < o, succ a < o) : NoMaxOrder o.toType :=
  ⟨has_succ_of_type_succ_lt (type_toType _ ▸ ho)⟩


@[deprecated toType_noMax_of_succ_lt (since := "2024-08-26")]
alias out_no_max_of_succ_lt := toType_noMax_of_succ_lt


theorem bounded_singleton {r : α → α → Prop} [IsWellOrder α r] (hr : (type r).IsLimit) (x) :
    Bounded r {x} := by
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : (Ordinal.type r).IsLimit
    x : α
    ⊢ Set.Bounded r (Singleton.singleton x)
  -/
  refine ⟨enum r ⟨succ (typein r x), hr.succ_lt (typein_lt_type r x)⟩, ?_⟩
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : (Ordinal.type r).IsLimit
    x : α
    ⊢ ∀ (b : α), Membership.mem (Singleton.singleton x) b → r b ((Ordinal.enum r)  …
  -/
  intro b hb
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : (Ordinal.type r).IsLimit
    x b : α
    hb : Membership.mem (Singleton.singleton x) b
    ⊢ r b ((Ordinal.enum r) ⟨Order.succ ((Ordinal.typein r).toRelEmbedding x), ⋯⟩)
  -/
  rw [mem_singleton_iff.1 hb]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : (Ordinal.type r).IsLimit
    x b : α
    hb : Membership.mem (Singleton.singleton x) b
    ⊢ r x ((Ordinal.enum r) ⟨Order.succ ((Ordinal.typein r).toRelEmbedding x), ⋯⟩)
  -/
  nth_rw 1 [← enum_typein r x]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : (Ordinal.type r).IsLimit
    x b : α
    hb : Membership.mem (Singleton.singleton x) b
    ⊢ r ((Ordinal.enum r) ⟨(Ordinal.typein r).toRelEmbedding x, ⋯⟩) ((Ordinal.enum …
  -/
  rw [@enum_lt_enum _ r, Subtype.mk_lt_mk]
  /-
    α : Type u_1
    r : α → α → Prop
    inst✝ : IsWellOrder α r
    hr : (Ordinal.type r).IsLimit
    x b : α
    hb : Membership.mem (Singleton.singleton x) b
    ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding x) (Order.succ ((Ordinal.typein r). …
  -/
  apply lt_succ
  /-
    🎉 no goals
  -/


@[simp]
theorem typein_ordinal (o : Ordinal.{u}) :
    @typein Ordinal (· < ·) _ o = Ordinal.lift.{u + 1} o := by
  /-
    o : Ordinal.{u}
    ⊢ Eq ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding o) (Ordinal.lif …
  -/
  refine Quotient.inductionOn o ?_
  /-
    o : Ordinal.{u}
    ⊢ ∀ (a : WellOrder), Eq ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedd …
  -/
  rintro ⟨α, r, wo⟩; apply Quotient.sound
  /-
    case mk.a
    o : Ordinal.{u}
    α : Type u
    r : α → α → Prop
    wo : IsWellOrder α r
    ⊢ HasEquiv.Equiv { α := Subtype fun b => (fun x1 x2 => LT.lt x1 x2) b (Quotien …
  -/
  constructor; refine ((RelIso.preimage Equiv.ulift r).trans (enum r).symm).symm
               /-
                 🎉 no goals
               -/

-- Porting note: `· < ·` requires a type ascription for an `IsWellOrder` instance.

@[deprecated typein_ordinal (since := "2024-09-19")]
theorem type_subrel_lt (o : Ordinal.{u}) :
    type (@Subrel Ordinal (· < ·) { o' : Ordinal | o' < o }) = Ordinal.lift.{u + 1} o :=
  typein_ordinal o


theorem mk_Iio_ordinal (o : Ordinal.{u}) :
    #(Iio o) = Cardinal.lift.{u + 1} o.card := by
  /-
    o : Ordinal.{u}
    ⊢ Eq (Cardinal.mk ↑(Set.Iio o)) (Cardinal.lift.{u + 1, u} o.card)
  -/
  rw [lift_card, ← typein_ordinal]
  /-
    o : Ordinal.{u}
    ⊢ Eq (Cardinal.mk ↑(Set.Iio o)) ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toR …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated mk_Iio_ordinal (since := "2024-09-19")]
theorem mk_initialSeg (o : Ordinal.{u}) :
    #{ o' : Ordinal | o' < o } = Cardinal.lift.{u + 1} o.card := mk_Iio_ordinal o



/-- A normal ordinal function is a strictly increasing function which is
  order-continuous, i.e., the image `f o` of a limit ordinal `o` is the sup of `f a` for
  `a < o`. -/
def IsNormal (f : Ordinal → Ordinal) : Prop :=
  (∀ o, f o < f (succ o)) ∧ ∀ o, IsLimit o → ∀ a, f o ≤ a ↔ ∀ b < o, f b ≤ a


theorem IsNormal.limit_le {f} (H : IsNormal f) :
    ∀ {o}, IsLimit o → ∀ {a}, f o ≤ a ↔ ∀ b < o, f b ≤ a :=
  @H.2


theorem IsNormal.limit_lt {f} (H : IsNormal f) {o} (h : IsLimit o) {a} :
    a < f o ↔ ∃ b < o, a < f b :=
                      /-
                        f : Ordinal.{u_4} → Ordinal.{u_5}
                        H : Ordinal.IsNormal f
                        o : Ordinal.{u_4}
                        h : o.IsLimit
                        a : Ordinal.{u_5}
                        ⊢ Iff (Not (LT.lt a (f o))) (Not (Exists fun b => And (LT.lt b o) (LT.lt a (f  …
                      -/
  not_iff_not.1 <| by simpa only [exists_prop, not_exists, not_and, not_lt] using H.2 _ h a
                      /-
                        🎉 no goals
                      -/


theorem IsNormal.strictMono {f} (H : IsNormal f) : StrictMono f := fun a b =>
  limitRecOn b (Not.elim (not_lt_of_le <| Ordinal.zero_le _))
    (fun _b IH h =>
      (lt_or_eq_of_le (le_of_lt_succ h)).elim (fun h => (IH h).trans (H.1 _)) fun e => e ▸ H.1 _)
    fun _b l _IH h => lt_of_lt_of_le (H.1 a) ((H.2 _ l _).1 le_rfl _ (l.succ_lt h))


theorem IsNormal.monotone {f} (H : IsNormal f) : Monotone f :=
  H.strictMono.monotone


theorem isNormal_iff_strictMono_limit (f : Ordinal → Ordinal) :
    IsNormal f ↔ StrictMono f ∧ ∀ o, IsLimit o → ∀ a, (∀ b < o, f b ≤ a) → f o ≤ a :=
  ⟨fun hf => ⟨hf.strictMono, fun a ha c => (hf.2 a ha c).2⟩, fun ⟨hs, hl⟩ =>
    ⟨fun a => hs (lt_succ a), fun a ha c =>
      ⟨fun hac _b hba => ((hs hba).trans_le hac).le, hl a ha c⟩⟩⟩


theorem IsNormal.lt_iff {f} (H : IsNormal f) {a b} : f a < f b ↔ a < b :=
  StrictMono.lt_iff_lt <| H.strictMono


theorem IsNormal.le_iff {f} (H : IsNormal f) {a b} : f a ≤ f b ↔ a ≤ b :=
  le_iff_le_iff_lt_iff_lt.2 H.lt_iff


theorem IsNormal.inj {f} (H : IsNormal f) {a b} : f a = f b ↔ a = b := by
  /-
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    a b : Ordinal.{u_4}
    ⊢ Iff (Eq (f a) (f b)) (Eq a b)
  -/
  simp only [le_antisymm_iff, H.le_iff]
  /-
    🎉 no goals
  -/


theorem IsNormal.id_le {f} (H : IsNormal f) : id ≤ f :=
  H.strictMono.id_le


theorem IsNormal.le_apply {f} (H : IsNormal f) {a} : a ≤ f a :=
  H.strictMono.le_apply


@[deprecated IsNormal.le_apply (since := "2024-09-11")]
theorem IsNormal.self_le {f} (H : IsNormal f) (a) : a ≤ f a :=
  H.strictMono.le_apply


theorem IsNormal.le_iff_eq {f} (H : IsNormal f) {a} : f a ≤ a ↔ f a = a :=
  H.le_apply.le_iff_eq


theorem IsNormal.le_set {f o} (H : IsNormal f) (p : Set Ordinal) (p0 : p.Nonempty) (b)
    (H₂ : ∀ o, b ≤ o ↔ ∀ a ∈ p, a ≤ o) : f b ≤ o ↔ ∀ a ∈ p, f a ≤ o :=
  ⟨fun h _ pa => (H.le_iff.2 ((H₂ _).1 le_rfl _ pa)).trans h, fun h => by
    -- Porting note: `refine'` didn't work well so `induction` is used
    induction b using limitRecOn with
    | H₁ =>
      cases' p0 with x px
      have := Ordinal.le_zero.1 ((H₂ _).1 (Ordinal.zero_le _) _ px)
      rw [this] at px
      exact h _ px
    | H₂ S _ =>
      rcases not_forall₂.1 (mt (H₂ S).2 <| (lt_succ S).not_le) with ⟨a, h₁, h₂⟩
      exact (H.le_iff.2 <| succ_le_of_lt <| not_le.1 h₂).trans (h _ h₁)
    | H₃ S L _ =>
      refine (H.2 _ L _).2 fun a h' => ?_
      rcases not_forall₂.1 (mt (H₂ a).2 h'.not_le) with ⟨b, h₁, h₂⟩
      exact (H.le_iff.2 <| (not_le.1 h₂).le).trans (h _ h₁)⟩


theorem IsNormal.le_set' {f o} (H : IsNormal f) (p : Set α) (p0 : p.Nonempty) (g : α → Ordinal) (b)
    (H₂ : ∀ o, b ≤ o ↔ ∀ a ∈ p, g a ≤ o) : f b ≤ o ↔ ∀ a ∈ p, f (g a) ≤ o := by
  /-
    α : Type u_1
    f : Ordinal.{u_4} → Ordinal.{u_5}
    o : Ordinal.{u_5}
    H : Ordinal.IsNormal f
    p : Set α
    p0 : p.Nonempty
    g : α → Ordinal.{u_4}
    b : Ordinal.{u_4}
    H₂ : ∀ (o : Ordinal.{u_4}), Iff (LE.le b o) (∀ (a : α), Membership.mem p a → L …
    ⊢ Iff (LE.le (f b) o) (∀ (a : α), Membership.mem p a → LE.le (f (g a)) o)
  -/
  simpa [H₂] using H.le_set (g '' p) (p0.image g) b
  /-
    🎉 no goals
  -/


theorem IsNormal.refl : IsNormal id :=
  ⟨lt_succ, fun _o l _a => Ordinal.limit_le l⟩


theorem IsNormal.trans {f g} (H₁ : IsNormal f) (H₂ : IsNormal g) : IsNormal (f ∘ g) :=
  ⟨fun _x => H₁.lt_iff.2 (H₂.1 _), fun o l _a =>
    H₁.le_set' (· < o) ⟨0, l.pos⟩ g _ fun _c => H₂.2 _ l _⟩


theorem IsNormal.isLimit {f} (H : IsNormal f) {o} (ho : IsLimit o) : IsLimit (f o) := by
  /-
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    ⊢ (f o).IsLimit
  -/
  rw [isLimit_iff, isSuccPrelimit_iff_succ_lt]
  /-
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    ⊢ And (Ne (f o) 0) (∀ (a : Ordinal.{u_5}), LT.lt a (f o) → LT.lt (Order.succ a …
  -/
  use (H.lt_iff.2 ho.pos).ne_bot
  /-
    case right
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    ⊢ ∀ (a : Ordinal.{u_5}), LT.lt a (f o) → LT.lt (Order.succ a) (f o)
  -/
  intro a ha
  /-
    case right
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    a : Ordinal.{u_5}
    ha : LT.lt a (f o)
    ⊢ LT.lt (Order.succ a) (f o)
  -/
  obtain ⟨b, hb, hab⟩ := (H.limit_lt ho).1 ha
  /-
    case right.intro.intro
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    a : Ordinal.{u_5}
    ha : LT.lt a (f o)
    b : Ordinal.{u_4}
    hb : LT.lt b o
    hab : LT.lt a (f b)
    ⊢ LT.lt (Order.succ a) (f o)
  -/
  rw [← succ_le_iff] at hab
  /-
    case right.intro.intro
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    a : Ordinal.{u_5}
    ha : LT.lt a (f o)
    b : Ordinal.{u_4}
    hb : LT.lt b o
    hab : LE.le (Order.succ a) (f b)
    ⊢ LT.lt (Order.succ a) (f o)
  -/
  apply hab.trans_lt
  /-
    case right.intro.intro
    f : Ordinal.{u_4} → Ordinal.{u_5}
    H : Ordinal.IsNormal f
    o : Ordinal.{u_4}
    ho : o.IsLimit
    a : Ordinal.{u_5}
    ha : LT.lt a (f o)
    b : Ordinal.{u_4}
    hb : LT.lt b o
    hab : LE.le (Order.succ a) (f b)
    ⊢ LT.lt (f b) (f o)
  -/
  rwa [H.lt_iff]
  /-
    🎉 no goals
  -/


private theorem add_le_of_limit {a b c : Ordinal} (h : IsLimit b) :
    a + b ≤ c ↔ ∀ b' < b, a + b' ≤ c :=
  ⟨fun h _ l => (add_le_add_left l.le _).trans h, fun H =>
    le_of_not_lt <| by
      -- Porting note: `induction` tactics are required because of the parser bug.
      induction a using inductionOn with
      | H α r =>
        induction b using inductionOn with
        | H β s =>
          intro l
          suffices ∀ x : β, Sum.Lex r s (Sum.inr x) (enum _ ⟨_, l⟩) by
            -- Porting note: `revert` & `intro` is required because `cases'` doesn't replace
            --               `enum _ _ l` in `this`.
            revert this; cases' enum _ ⟨_, l⟩ with x x <;> intro this
            · cases this (enum s ⟨0, h.pos⟩)
            · exact irrefl _ (this _)
          intro x
          rw [← typein_lt_typein (Sum.Lex r s), typein_enum]
          have := H _ (h.succ_lt (typein_lt_type s x))
          rw [add_succ, succ_le_iff] at this
          refine
            (RelEmbedding.ofMonotone (fun a => ?_) fun a b => ?_).ordinal_type_le.trans_lt this
          · rcases a with ⟨a | b, h⟩
            · exact Sum.inl a
            · exact Sum.inr ⟨b, by cases h; assumption⟩
          · rcases a with ⟨a | a, h₁⟩ <;> rcases b with ⟨b | b, h₂⟩ <;> cases h₁ <;> cases h₂ <;>
              rintro ⟨⟩ <;> constructor <;> assumption⟩


theorem isNormal_add_right (a : Ordinal) : IsNormal (a + ·) :=
  ⟨fun b => (add_lt_add_iff_left a).2 (lt_succ b), fun _b l _c => add_le_of_limit l⟩


@[deprecated isNormal_add_right (since := "2024-10-11")]
alias add_isNormal := isNormal_add_right


theorem isLimit_add (a) {b} : IsLimit b → IsLimit (a + b) :=
  (isNormal_add_right a).isLimit


@[deprecated isLimit_add (since := "2024-10-11")]
alias add_isLimit := isLimit_add


alias IsLimit.add := add_isLimit


/-- The set in the definition of subtraction is nonempty. -/
private theorem sub_nonempty {a b : Ordinal} : { o | a ≤ b + o }.Nonempty :=
  ⟨a, le_add_left _ _⟩


/-- `a - b` is the unique ordinal satisfying `b + (a - b) = a` when `b ≤ a`. -/
instance sub : Sub Ordinal :=
  ⟨fun a b => sInf { o | a ≤ b + o }⟩


theorem le_add_sub (a b : Ordinal) : a ≤ b + (a - b) :=
  csInf_mem sub_nonempty


theorem sub_le {a b c : Ordinal} : a - b ≤ c ↔ a ≤ b + c :=
  ⟨fun h => (le_add_sub a b).trans (add_le_add_left h _), fun h => csInf_le' h⟩


theorem lt_sub {a b c : Ordinal} : a < b - c ↔ c + a < b :=
  lt_iff_lt_of_le_iff_le sub_le


theorem add_sub_cancel (a b : Ordinal) : a + b - a = b :=
  le_antisymm (sub_le.2 <| le_rfl) ((add_le_add_iff_left a).1 <| le_add_sub _ _)


theorem sub_eq_of_add_eq {a b c : Ordinal} (h : a + b = c) : c - a = b :=
  h ▸ add_sub_cancel _ _


theorem sub_le_self (a b : Ordinal) : a - b ≤ a :=
  sub_le.2 <| le_add_left _ _


protected theorem add_sub_cancel_of_le {a b : Ordinal} (h : b ≤ a) : b + (a - b) = a :=
  (le_add_sub a b).antisymm'
    (by
      /-
        a b : Ordinal.{u_4}
        h : LE.le b a
        ⊢ LE.le (HAdd.hAdd b (HSub.hSub a b)) a
      -/
      rcases zero_or_succ_or_limit (a - b) with (e | ⟨c, e⟩ | l)
        /-
          case inl
          a b : Ordinal.{u_4}
          h : LE.le b a
          e : Eq (HSub.hSub a b) 0
          ⊢ LE.le (HAdd.hAdd b (HSub.hSub a b)) a
        -/
      · simp only [e, add_zero, h]
        /-
          🎉 no goals
        -/
        /-
          case inr.inl.intro
          a b : Ordinal.{u_4}
          h : LE.le b a
          c : Ordinal.{u_4}
          e : Eq (HSub.hSub a b) (Order.succ c)
          ⊢ LE.le (HAdd.hAdd b (HSub.hSub a b)) a
        -/
      · rw [e, add_succ, succ_le_iff, ← lt_sub, e]
        /-
          case inr.inl.intro
          a b : Ordinal.{u_4}
          h : LE.le b a
          c : Ordinal.{u_4}
          e : Eq (HSub.hSub a b) (Order.succ c)
          ⊢ LT.lt c (Order.succ c)
        -/
        exact lt_succ c
        /-
          🎉 no goals
        -/
        /-
          case inr.inr
          a b : Ordinal.{u_4}
          h : LE.le b a
          l : (HSub.hSub a b).IsLimit
          ⊢ LE.le (HAdd.hAdd b (HSub.hSub a b)) a
        -/
      · exact (add_le_of_limit l).2 fun c l => (lt_sub.1 l).le)
        /-
          🎉 no goals
        -/


theorem le_sub_of_le {a b c : Ordinal} (h : b ≤ a) : c ≤ a - b ↔ b + c ≤ a := by
  /-
    a b c : Ordinal.{u_4}
    h : LE.le b a
    ⊢ Iff (LE.le c (HSub.hSub a b)) (LE.le (HAdd.hAdd b c) a)
  -/
  rw [← add_le_add_iff_left b, Ordinal.add_sub_cancel_of_le h]
  /-
    🎉 no goals
  -/


theorem sub_lt_of_le {a b c : Ordinal} (h : b ≤ a) : a - b < c ↔ a < b + c :=
  lt_iff_lt_of_le_iff_le (le_sub_of_le h)


instance existsAddOfLE : ExistsAddOfLE Ordinal :=
  ⟨fun h => ⟨_, (Ordinal.add_sub_cancel_of_le h).symm⟩⟩


@[simp]
                                                 /-
                                                   a : Ordinal.{u_4}
                                                   ⊢ Eq (HSub.hSub a 0) a
                                                 -/
theorem sub_zero (a : Ordinal) : a - 0 = a := by simpa only [zero_add] using add_sub_cancel 0 a
                                                 /-
                                                   🎉 no goals
                                                 -/


@[simp]
                                                 /-
                                                   a : Ordinal.{u_4}
                                                   ⊢ Eq (HSub.hSub 0 a) 0
                                                 -/
theorem zero_sub (a : Ordinal) : 0 - a = 0 := by rw [← Ordinal.le_zero]; apply sub_le_self
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
                                                 /-
                                                   a : Ordinal.{u_4}
                                                   ⊢ Eq (HSub.hSub a a) 0
                                                 -/
theorem sub_self (a : Ordinal) : a - a = 0 := by simpa only [add_zero] using add_sub_cancel a 0
                                                 /-
                                                   🎉 no goals
                                                 -/


protected theorem sub_eq_zero_iff_le {a b : Ordinal} : a - b = 0 ↔ a ≤ b :=
               /-
                 a b : Ordinal.{u_4}
                 h : Eq (HSub.hSub a b) 0
                 ⊢ LE.le a b
               -/
  ⟨fun h => by simpa only [h, add_zero] using le_add_sub a b, fun h => by
               /-
                 🎉 no goals
               -/
    /-
      a b : Ordinal.{u_4}
      h : LE.le a b
      ⊢ Eq (HSub.hSub a b) 0
    -/
    rwa [← Ordinal.le_zero, sub_le, add_zero]⟩
    /-
      🎉 no goals
    -/


protected theorem sub_ne_zero_iff_lt {a b : Ordinal} : a - b ≠ 0 ↔ b < a := by
  /-
    a b : Ordinal.{u_4}
    ⊢ Iff (Ne (HSub.hSub a b) 0) (LT.lt b a)
  -/
  simpa using Ordinal.sub_eq_zero_iff_le.not
  /-
    🎉 no goals
  -/


theorem sub_sub (a b c : Ordinal) : a - b - c = a - (b + c) :=
                                  /-
                                    a b c d : Ordinal.{u_4}
                                    ⊢ Iff (LE.le (HSub.hSub (HSub.hSub a b) c) d) (LE.le (HSub.hSub a (HAdd.hAdd b …
                                  -/
  eq_of_forall_ge_iff fun d => by rw [sub_le, sub_le, sub_le, add_assoc]
                                  /-
                                    🎉 no goals
                                  -/


@[simp]
theorem add_sub_add_cancel (a b c : Ordinal) : a + b - (a + c) = b - c := by
  /-
    a b c : Ordinal.{u_4}
    ⊢ Eq (HSub.hSub (HAdd.hAdd a b) (HAdd.hAdd a c)) (HSub.hSub b c)
  -/
  rw [← sub_sub, add_sub_cancel]
  /-
    🎉 no goals
  -/


theorem le_sub_of_add_le {a b c : Ordinal} (h : b + c ≤ a) : c ≤ a - b := by
  /-
    a b c : Ordinal.{u_4}
    h : LE.le (HAdd.hAdd b c) a
    ⊢ LE.le c (HSub.hSub a b)
  -/
  rw [← add_le_add_iff_left b]
  /-
    a b c : Ordinal.{u_4}
    h : LE.le (HAdd.hAdd b c) a
    ⊢ LE.le (HAdd.hAdd b c) (HAdd.hAdd b (HSub.hSub a b))
  -/
  exact h.trans (le_add_sub a b)
  /-
    🎉 no goals
  -/


theorem sub_lt_of_lt_add {a b c : Ordinal} (h : a < b + c) (hc : 0 < c) : a - b < c := by
  /-
    a b c : Ordinal.{u_4}
    h : LT.lt a (HAdd.hAdd b c)
    hc : LT.lt 0 c
    ⊢ LT.lt (HSub.hSub a b) c
  -/
  obtain hab | hba := lt_or_le a b
    /-
      case inl
      a b c : Ordinal.{u_4}
      h : LT.lt a (HAdd.hAdd b c)
      hc : LT.lt 0 c
      hab : LT.lt a b
      ⊢ LT.lt (HSub.hSub a b) c
    -/
  · rwa [Ordinal.sub_eq_zero_iff_le.2 hab.le]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b c : Ordinal.{u_4}
      h : LT.lt a (HAdd.hAdd b c)
      hc : LT.lt 0 c
      hba : LE.le b a
      ⊢ LT.lt (HSub.hSub a b) c
    -/
  · rwa [sub_lt_of_le hba]
    /-
      🎉 no goals
    -/


theorem lt_add_iff {a b c : Ordinal} (hc : c ≠ 0) : a < b + c ↔ ∃ d < c, a ≤ b + d := by
  /-
    a b c : Ordinal.{u_4}
    hc : Ne c 0
    ⊢ Iff (LT.lt a (HAdd.hAdd b c)) (Exists fun d => And (LT.lt d c) (LE.le a (HAd …
  -/
  use fun h ↦ ⟨_, sub_lt_of_lt_add h hc.bot_lt, le_add_sub a b⟩
  /-
    case mpr
    a b c : Ordinal.{u_4}
    hc : Ne c 0
    ⊢ (Exists fun d => And (LT.lt d c) (LE.le a (HAdd.hAdd b d))) → LT.lt a (HAdd. …
  -/
  rintro ⟨d, hd, ha⟩
  /-
    case mpr.intro.intro
    a b c : Ordinal.{u_4}
    hc : Ne c 0
    d : Ordinal.{u_4}
    hd : LT.lt d c
    ha : LE.le a (HAdd.hAdd b d)
    ⊢ LT.lt a (HAdd.hAdd b c)
  -/
  exact ha.trans_lt (add_lt_add_left hd b)
  /-
    🎉 no goals
  -/


theorem add_le_iff {a b c : Ordinal} (hb : b ≠ 0) : a + b ≤ c ↔ ∀ d < b, a + d < c := by
  /-
    a b c : Ordinal.{u_4}
    hb : Ne b 0
    ⊢ Iff (LE.le (HAdd.hAdd a b) c) (∀ (d : Ordinal.{u_4}), LT.lt d b → LT.lt (HAd …
  -/
  simpa using (lt_add_iff hb).not
  /-
    🎉 no goals
  -/


@[deprecated add_le_iff (since := "2024-12-08")]
theorem add_le_of_forall_add_lt {a b c : Ordinal} (hb : 0 < b) (h : ∀ d < b, a + d < c) :
    a + b ≤ c :=
  (add_le_iff hb.ne').2 h


theorem isLimit_sub {a b} (ha : IsLimit a) (h : b < a) : IsLimit (a - b) := by
  /-
    a b : Ordinal.{u_4}
    ha : a.IsLimit
    h : LT.lt b a
    ⊢ (HSub.hSub a b).IsLimit
  -/
  rw [isLimit_iff, Ordinal.sub_ne_zero_iff_lt, isSuccPrelimit_iff_succ_lt]
  /-
    a b : Ordinal.{u_4}
    ha : a.IsLimit
    h : LT.lt b a
    ⊢ And (LT.lt b a) (∀ (a_1 : Ordinal.{u_4}), LT.lt a_1 (HSub.hSub a b) → LT.lt  …
  -/
  refine ⟨h, fun c hc ↦ ?_⟩
  /-
    a b : Ordinal.{u_4}
    ha : a.IsLimit
    h : LT.lt b a
    c : Ordinal.{u_4}
    hc : LT.lt c (HSub.hSub a b)
    ⊢ LT.lt (Order.succ c) (HSub.hSub a b)
  -/
  rw [lt_sub] at hc ⊢
  /-
    a b : Ordinal.{u_4}
    ha : a.IsLimit
    h : LT.lt b a
    c : Ordinal.{u_4}
    hc : LT.lt (HAdd.hAdd b c) a
    ⊢ LT.lt (HAdd.hAdd b (Order.succ c)) a
  -/
  rw [add_succ]
  /-
    a b : Ordinal.{u_4}
    ha : a.IsLimit
    h : LT.lt b a
    c : Ordinal.{u_4}
    hc : LT.lt (HAdd.hAdd b c) a
    ⊢ LT.lt (Order.succ (HAdd.hAdd b c)) a
  -/
  exact ha.succ_lt hc
  /-
    🎉 no goals
  -/


@[deprecated isLimit_sub (since := "2024-10-11")]
alias sub_isLimit := isLimit_sub


/-- The multiplication of ordinals `o₁` and `o₂` is the (well founded) lexicographic order on
`o₂ × o₁`. -/
instance monoid : Monoid Ordinal.{u} where
  mul a b :=
    Quotient.liftOn₂ a b
      (fun ⟨α, r, _⟩ ⟨β, s, _⟩ => ⟦⟨β × α, Prod.Lex s r, inferInstance⟩⟧ :
        WellOrder → WellOrder → Ordinal)
      fun ⟨_, _, _⟩ _ _ _ ⟨f⟩ ⟨g⟩ => Quot.sound ⟨RelIso.prodLexCongr g f⟩
  one := 1
  mul_assoc a b c :=
    Quotient.inductionOn₃ a b c fun ⟨α, r, _⟩ ⟨β, s, _⟩ ⟨γ, t, _⟩ =>
      Eq.symm <|
        Quotient.sound
          ⟨⟨prodAssoc _ _ _, @fun a b => by
              /-
                α✝ : Type u_1
                β✝ : Type u_2
                γ✝ : Type u_3
                r✝ : α✝ → α✝ → Prop
                s✝ : β✝ → β✝ → Prop
                t✝ : γ✝ → γ✝ → Prop
                a✝ b✝ c : Ordinal.{u}
                x✝² x✝¹ x✝ : WellOrder
                α : Type u
                r : α → α → Prop
                wo✝² : IsWellOrder α r
                β : Type u
                s : β → β → Prop
                wo✝¹ : IsWellOrder β s
                γ : Type u
                t : γ → γ → Prop
                wo✝ : IsWellOrder γ t
                a b : Prod (Prod γ β) α
                ⊢ Iff (Prod.Lex t (Prod.Lex s r) ((Equiv.prodAssoc γ β α) a) ((Equiv.prodAssoc …
              -/
              rcases a with ⟨⟨a₁, a₂⟩, a₃⟩
              /-
                case mk.mk
                α✝ : Type u_1
                β✝ : Type u_2
                γ✝ : Type u_3
                r✝ : α✝ → α✝ → Prop
                s✝ : β✝ → β✝ → Prop
                t✝ : γ✝ → γ✝ → Prop
                a b✝ c : Ordinal.{u}
                x✝² x✝¹ x✝ : WellOrder
                α : Type u
                r : α → α → Prop
                wo✝² : IsWellOrder α r
                β : Type u
                s : β → β → Prop
                wo✝¹ : IsWellOrder β s
                γ : Type u
                t : γ → γ → Prop
                wo✝ : IsWellOrder γ t
                b : Prod (Prod γ β) α
                a₃ : α
                a₁ : γ
                a₂ : β
                ⊢ Iff (Prod.Lex t (Prod.Lex s r) ((Equiv.prodAssoc γ β α) { fst := { fst := a₁ …
              -/
              rcases b with ⟨⟨b₁, b₂⟩, b₃⟩
              /-
                case mk.mk.mk.mk
                α✝ : Type u_1
                β✝ : Type u_2
                γ✝ : Type u_3
                r✝ : α✝ → α✝ → Prop
                s✝ : β✝ → β✝ → Prop
                t✝ : γ✝ → γ✝ → Prop
                a b c : Ordinal.{u}
                x✝² x✝¹ x✝ : WellOrder
                α : Type u
                r : α → α → Prop
                wo✝² : IsWellOrder α r
                β : Type u
                s : β → β → Prop
                wo✝¹ : IsWellOrder β s
                γ : Type u
                t : γ → γ → Prop
                wo✝ : IsWellOrder γ t
                a₃ : α
                a₁ : γ
                a₂ : β
                b₃ : α
                b₁ : γ
                b₂ : β
                ⊢ Iff (Prod.Lex t (Prod.Lex s r) ((Equiv.prodAssoc γ β α) { fst := { fst := a₁ …
              -/
              simp [Prod.lex_def, and_or_left, or_assoc, and_assoc]⟩⟩
              /-
                🎉 no goals
              -/
  mul_one a :=
    inductionOn a fun α r _ =>
      Quotient.sound
        ⟨⟨punitProd _, @fun a b => by
            /-
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : Prod PUnit.{u + 1} α
              ⊢ Iff (r ((Equiv.punitProd α) a) ((Equiv.punitProd α) b)) (Prod.Lex EmptyRelat …
            -/
            rcases a with ⟨⟨⟨⟩⟩, a⟩; rcases b with ⟨⟨⟨⟩⟩, b⟩
            /-
              case mk.unit.mk.unit
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : α
              ⊢ Iff (r ((Equiv.punitProd α) { fst := PUnit.unit, snd := a }) ((Equiv.punitPr …
            -/
            simp only [Prod.lex_def, EmptyRelation, false_or]
            /-
              case mk.unit.mk.unit
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : α
              ⊢ Iff (r ((Equiv.punitProd α) { fst := PUnit.unit, snd := a }) ((Equiv.punitPr …
            -/
            simp only [eq_self_iff_true, true_and]
            /-
              case mk.unit.mk.unit
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : α
              ⊢ Iff (r ((Equiv.punitProd α) { fst := PUnit.unit, snd := a }) ((Equiv.punitPr …
            -/
            rfl⟩⟩
            /-
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : Prod α PUnit.{u + 1}
              ⊢ Iff (r ((Equiv.prodPUnit α) a) ((Equiv.prodPUnit α) b)) (Prod.Lex r EmptyRel …
            -/
            /-
              🎉 no goals
            -/
            /-
              case mk.unit.mk.unit
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : α
              ⊢ Iff (r ((Equiv.prodPUnit α) { fst := a, snd := PUnit.unit }) ((Equiv.prodPUn …
            -/
  one_mul a :=
            /-
              case mk.unit.mk.unit
              α✝ : Type u_1
              β : Type u_2
              γ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s : β → β → Prop
              t : γ → γ → Prop
              a✝ : Ordinal.{u}
              α : Type u
              r : α → α → Prop
              x✝ : IsWellOrder α r
              a b : α
              ⊢ Iff (r ((Equiv.prodPUnit α) { fst := a, snd := PUnit.unit }) ((Equiv.prodPUn …
            -/
    inductionOn a fun α r _ =>
            /-
              🎉 no goals
            -/
      Quotient.sound
        ⟨⟨prodPUnit _, @fun a b => by
            rcases a with ⟨a, ⟨⟨⟩⟩⟩; rcases b with ⟨b, ⟨⟨⟩⟩⟩
            simp only [Prod.lex_def, EmptyRelation, and_false, or_false]
            rfl⟩⟩


@[simp]
theorem type_prod_lex {α β : Type u} (r : α → α → Prop) (s : β → β → Prop) [IsWellOrder α r]
    [IsWellOrder β s] : type (Prod.Lex s r) = type r * type s :=
  rfl


private theorem mul_eq_zero' {a b : Ordinal} : a * b = 0 ↔ a = 0 ∨ b = 0 :=
  inductionOn a fun α _ _ =>
    inductionOn b fun β _ _ => by
      /-
        a b : Ordinal.{u_4}
        α : Type u_4
        x✝³ : α → α → Prop
        x✝² : IsWellOrder α x✝³
        β : Type u_4
        x✝¹ : β → β → Prop
        x✝ : IsWellOrder β x✝¹
        ⊢ Iff (Eq (HMul.hMul (Ordinal.type x✝³) (Ordinal.type x✝¹)) 0) (Or (Eq (Ordina …
      -/
      simp_rw [← type_prod_lex, type_eq_zero_iff_isEmpty]
      /-
        a b : Ordinal.{u_4}
        α : Type u_4
        x✝³ : α → α → Prop
        x✝² : IsWellOrder α x✝³
        β : Type u_4
        x✝¹ : β → β → Prop
        x✝ : IsWellOrder β x✝¹
        ⊢ Iff (IsEmpty (Prod β α)) (Or (IsEmpty α) (IsEmpty β))
      -/
      rw [or_comm]
      /-
        a b : Ordinal.{u_4}
        α : Type u_4
        x✝³ : α → α → Prop
        x✝² : IsWellOrder α x✝³
        β : Type u_4
        x✝¹ : β → β → Prop
        x✝ : IsWellOrder β x✝¹
        ⊢ Iff (IsEmpty (Prod β α)) (Or (IsEmpty β) (IsEmpty α))
      -/
      exact isEmpty_prod
      /-
        🎉 no goals
      -/


instance monoidWithZero : MonoidWithZero Ordinal :=
  { Ordinal.monoid with
    zero := 0
    mul_zero := fun _a => mul_eq_zero'.2 <| Or.inr rfl
    zero_mul := fun _a => mul_eq_zero'.2 <| Or.inl rfl }


instance noZeroDivisors : NoZeroDivisors Ordinal :=
  ⟨fun {_ _} => mul_eq_zero'.1⟩


@[simp]
theorem lift_mul (a b : Ordinal.{v}) : lift.{u} (a * b) = lift.{u} a * lift.{u} b :=
  Quotient.inductionOn₂ a b fun ⟨_α, _r, _⟩ ⟨_β, _s, _⟩ =>
    Quotient.sound
      ⟨(RelIso.preimage Equiv.ulift _).trans
          (RelIso.prodLexCongr (RelIso.preimage Equiv.ulift _)
              (RelIso.preimage Equiv.ulift _)).symm⟩


@[simp]
theorem card_mul (a b) : card (a * b) = card a * card b :=
  Quotient.inductionOn₂ a b fun ⟨α, _r, _⟩ ⟨β, _s, _⟩ => mul_comm #β #α


instance leftDistribClass : LeftDistribClass Ordinal.{u} :=
  ⟨fun a b c =>
    Quotient.inductionOn₃ a b c fun ⟨α, r, _⟩ ⟨β, s, _⟩ ⟨γ, t, _⟩ =>
      Quotient.sound
        ⟨⟨sumProdDistrib _ _ _, by
          /-
            α✝ : Type u_1
            β✝ : Type u_2
            γ✝ : Type u_3
            r✝ : α✝ → α✝ → Prop
            s✝ : β✝ → β✝ → Prop
            t✝ : γ✝ → γ✝ → Prop
            a b c : Ordinal.{u}
            x✝² x✝¹ x✝ : WellOrder
            α : Type u
            r : α → α → Prop
            wo✝² : IsWellOrder α r
            β : Type u
            s : β → β → Prop
            wo✝¹ : IsWellOrder β s
            γ : Type u
            t : γ → γ → Prop
            wo✝ : IsWellOrder γ t
            ⊢ ∀ {a b : Prod (Sum β γ) α}, Iff (Sum.Lex (Prod.Lex s r) (Prod.Lex t r) ((Equ …
          -/
          rintro ⟨a₁ | a₁, a₂⟩ ⟨b₁ | b₁, b₂⟩ <;>
            simp only [Prod.lex_def, Sum.lex_inl_inl, Sum.Lex.sep, Sum.lex_inr_inl, Sum.lex_inr_inr,
              sumProdDistrib_apply_left, sumProdDistrib_apply_right, reduceCtorEq] <;>
            -- Porting note: `Sum.inr.inj_iff` is required.
            /-
              case mk.inl.mk.inl
              α✝ : Type u_1
              β✝ : Type u_2
              γ✝ : Type u_3
              r✝ : α✝ → α✝ → Prop
              s✝ : β✝ → β✝ → Prop
              t✝ : γ✝ → γ✝ → Prop
              a b c : Ordinal.{u}
              x✝² x✝¹ x✝ : WellOrder
              α : Type u
              r : α → α → Prop
              wo✝² : IsWellOrder α r
              β : Type u
              s : β → β → Prop
              wo✝¹ : IsWellOrder β s
              γ : Type u
              t : γ → γ → Prop
              wo✝ : IsWellOrder γ t
              a₂ : α
              a₁ : β
              b₂ : α
              b₁ : β
              ⊢ Iff (Or (s a₁ b₁) (And (Eq a₁ b₁) (r a₂ b₂))) (Or (s a₁ b₁) (And (Eq (Sum.in …
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
            simp only [Sum.inl.inj_iff, Sum.inr.inj_iff, true_or, false_and, false_or]⟩⟩⟩
            /-
              🎉 no goals
            -/


theorem mul_succ (a b : Ordinal) : a * succ b = a * b + a :=
  mul_add_one a b


instance mulLeftMono : MulLeftMono Ordinal.{u} :=
  ⟨fun c a b =>
    Quotient.inductionOn₃ a b c fun ⟨α, r, _⟩ ⟨β, s, _⟩ ⟨γ, t, _⟩ ⟨f⟩ => by
      refine
        (RelEmbedding.ofMonotone (fun a : α × γ => (f a.1, a.2)) fun a b h => ?_).ordinal_type_le
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        r✝ : α✝ → α✝ → Prop
        s✝ : β✝ → β✝ → Prop
        t✝ : γ✝ → γ✝ → Prop
        c a✝ b✝ : Ordinal.{u}
        x✝³ x✝² x✝¹ : WellOrder
        α : Type u
        r : α → α → Prop
        wo✝² : IsWellOrder α r
        β : Type u
        s : β → β → Prop
        wo✝¹ : IsWellOrder β s
        x✝ : LE.le (Quotient.mk Ordinal.isEquivalent { α := α, r := r, wo := wo✝² }) ( …
        γ : Type u
        t : γ → γ → Prop
        wo✝ : IsWellOrder γ t
        f : InitialSeg r s
        a b : Prod α γ
        h : Prod.Lex r t a b
        ⊢ Prod.Lex s t ((fun a => { fst := f a.1, snd := a.2 }) a) ((fun a => { fst := …
      -/
      cases' h with a₁ b₁ a₂ b₂ h' a b₁ b₂ h'
        /-
          case left
          α✝ : Type u_1
          β✝ : Type u_2
          γ✝ : Type u_3
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t✝ : γ✝ → γ✝ → Prop
          c a b : Ordinal.{u}
          x✝³ x✝² x✝¹ : WellOrder
          α : Type u
          r : α → α → Prop
          wo✝² : IsWellOrder α r
          β : Type u
          s : β → β → Prop
          wo✝¹ : IsWellOrder β s
          x✝ : LE.le (Quotient.mk Ordinal.isEquivalent { α := α, r := r, wo := wo✝² }) ( …
          γ : Type u
          t : γ → γ → Prop
          wo✝ : IsWellOrder γ t
          f : InitialSeg r s
          a₁ : α
          b₁ : γ
          a₂ : α
          b₂ : γ
          h' : r a₁ a₂
          ⊢ Prod.Lex s t ((fun a => { fst := f a.1, snd := a.2 }) { fst := a₁, snd := b₁ …
        -/
      · exact Prod.Lex.left _ _ (f.toRelEmbedding.map_rel_iff.2 h')
        /-
          🎉 no goals
        -/
        /-
          case right
          α✝ : Type u_1
          β✝ : Type u_2
          γ✝ : Type u_3
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t✝ : γ✝ → γ✝ → Prop
          c a✝ b : Ordinal.{u}
          x✝³ x✝² x✝¹ : WellOrder
          α : Type u
          r : α → α → Prop
          wo✝² : IsWellOrder α r
          β : Type u
          s : β → β → Prop
          wo✝¹ : IsWellOrder β s
          x✝ : LE.le (Quotient.mk Ordinal.isEquivalent { α := α, r := r, wo := wo✝² }) ( …
          γ : Type u
          t : γ → γ → Prop
          wo✝ : IsWellOrder γ t
          f : InitialSeg r s
          a : α
          b₁ b₂ : γ
          h' : t b₁ b₂
          ⊢ Prod.Lex s t ((fun a => { fst := f a.1, snd := a.2 }) { fst := a, snd := b₁  …
        -/
      · exact Prod.Lex.right _ h'⟩
        /-
          🎉 no goals
        -/


instance mulRightMono : MulRightMono Ordinal.{u} :=
  ⟨fun c a b =>
    Quotient.inductionOn₃ a b c fun ⟨α, r, _⟩ ⟨β, s, _⟩ ⟨γ, t, _⟩ ⟨f⟩ => by
      refine
        (RelEmbedding.ofMonotone (fun a : γ × α => (a.1, f a.2)) fun a b h => ?_).ordinal_type_le
      /-
        α✝ : Type u_1
        β✝ : Type u_2
        γ✝ : Type u_3
        r✝ : α✝ → α✝ → Prop
        s✝ : β✝ → β✝ → Prop
        t✝ : γ✝ → γ✝ → Prop
        c a✝ b✝ : Ordinal.{u}
        x✝³ x✝² x✝¹ : WellOrder
        α : Type u
        r : α → α → Prop
        wo✝² : IsWellOrder α r
        β : Type u
        s : β → β → Prop
        wo✝¹ : IsWellOrder β s
        x✝ : LE.le (Quotient.mk Ordinal.isEquivalent { α := α, r := r, wo := wo✝² }) ( …
        γ : Type u
        t : γ → γ → Prop
        wo✝ : IsWellOrder γ t
        f : InitialSeg r s
        a b : Prod γ α
        h : Prod.Lex t r a b
        ⊢ Prod.Lex t s ((fun a => { fst := a.1, snd := f a.2 }) a) ((fun a => { fst := …
      -/
      cases' h with a₁ b₁ a₂ b₂ h' a b₁ b₂ h'
        /-
          case left
          α✝ : Type u_1
          β✝ : Type u_2
          γ✝ : Type u_3
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t✝ : γ✝ → γ✝ → Prop
          c a b : Ordinal.{u}
          x✝³ x✝² x✝¹ : WellOrder
          α : Type u
          r : α → α → Prop
          wo✝² : IsWellOrder α r
          β : Type u
          s : β → β → Prop
          wo✝¹ : IsWellOrder β s
          x✝ : LE.le (Quotient.mk Ordinal.isEquivalent { α := α, r := r, wo := wo✝² }) ( …
          γ : Type u
          t : γ → γ → Prop
          wo✝ : IsWellOrder γ t
          f : InitialSeg r s
          a₁ : γ
          b₁ : α
          a₂ : γ
          b₂ : α
          h' : t a₁ a₂
          ⊢ Prod.Lex t s ((fun a => { fst := a.1, snd := f a.2 }) { fst := a₁, snd := b₁ …
        -/
      · exact Prod.Lex.left _ _ h'
        /-
          🎉 no goals
        -/
        /-
          case right
          α✝ : Type u_1
          β✝ : Type u_2
          γ✝ : Type u_3
          r✝ : α✝ → α✝ → Prop
          s✝ : β✝ → β✝ → Prop
          t✝ : γ✝ → γ✝ → Prop
          c a✝ b : Ordinal.{u}
          x✝³ x✝² x✝¹ : WellOrder
          α : Type u
          r : α → α → Prop
          wo✝² : IsWellOrder α r
          β : Type u
          s : β → β → Prop
          wo✝¹ : IsWellOrder β s
          x✝ : LE.le (Quotient.mk Ordinal.isEquivalent { α := α, r := r, wo := wo✝² }) ( …
          γ : Type u
          t : γ → γ → Prop
          wo✝ : IsWellOrder γ t
          f : InitialSeg r s
          a : γ
          b₁ b₂ : α
          h' : r b₁ b₂
          ⊢ Prod.Lex t s ((fun a => { fst := a.1, snd := f a.2 }) { fst := a, snd := b₁  …
        -/
      · exact Prod.Lex.right _ (f.toRelEmbedding.map_rel_iff.2 h')⟩
        /-
          🎉 no goals
        -/


theorem le_mul_left (a : Ordinal) {b : Ordinal} (hb : 0 < b) : a ≤ a * b := by
  /-
    a b : Ordinal.{u_4}
    hb : LT.lt 0 b
    ⊢ LE.le a (HMul.hMul a b)
  -/
  convert mul_le_mul_left' (one_le_iff_pos.2 hb) a
  /-
    case h.e'_3
    a b : Ordinal.{u_4}
    hb : LT.lt 0 b
    ⊢ Eq a (HMul.hMul a 1)
  -/
  rw [mul_one a]
  /-
    🎉 no goals
  -/


theorem le_mul_right (a : Ordinal) {b : Ordinal} (hb : 0 < b) : a ≤ b * a := by
  /-
    a b : Ordinal.{u_4}
    hb : LT.lt 0 b
    ⊢ LE.le a (HMul.hMul b a)
  -/
  convert mul_le_mul_right' (one_le_iff_pos.2 hb) a
  /-
    case h.e'_3
    a b : Ordinal.{u_4}
    hb : LT.lt 0 b
    ⊢ Eq a (HMul.hMul 1 a)
  -/
  rw [one_mul a]
  /-
    🎉 no goals
  -/


private theorem mul_le_of_limit_aux {α β r s} [IsWellOrder α r] [IsWellOrder β s] {c}
    (h : IsLimit (type s)) (H : ∀ b' < type s, type r * b' ≤ c) (l : c < type r * type s) :
    False := by
  suffices ∀ a b, Prod.Lex s r (b, a) (enum _ ⟨_, l⟩) by
    cases' enum _ ⟨_, l⟩ with b a
    exact irrefl _ (this _ _)
  /-
    α β : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    c : Ordinal.{u_4}
    h : (Ordinal.type s).IsLimit
    H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
    l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
    ⊢ ∀ (a : α) (b : β), Prod.Lex s r { fst := b, snd := a } ((Ordinal.enum (Prod. …
  -/
  intro a b
  /-
    α β : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    c : Ordinal.{u_4}
    h : (Ordinal.type s).IsLimit
    H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
    l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
    a : α
    b : β
    ⊢ Prod.Lex s r { fst := b, snd := a } ((Ordinal.enum (Prod.Lex s r)) ⟨c, l⟩)
  -/
  rw [← typein_lt_typein (Prod.Lex s r), typein_enum]
  /-
    α β : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    c : Ordinal.{u_4}
    h : (Ordinal.type s).IsLimit
    H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
    l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
    a : α
    b : β
    ⊢ LT.lt ((Ordinal.typein (Prod.Lex s r)).toRelEmbedding { fst := b, snd := a } …
  -/
  have := H _ (h.succ_lt (typein_lt_type s b))
  /-
    α β : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    c : Ordinal.{u_4}
    h : (Ordinal.type s).IsLimit
    H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
    l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
    a : α
    b : β
    this : LE.le (HMul.hMul (Ordinal.type r) (Order.succ ((Ordinal.typein s).toRel …
    ⊢ LT.lt ((Ordinal.typein (Prod.Lex s r)).toRelEmbedding { fst := b, snd := a } …
  -/
  rw [mul_succ] at this
  /-
    α β : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    c : Ordinal.{u_4}
    h : (Ordinal.type s).IsLimit
    H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
    l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
    a : α
    b : β
    this : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
    ⊢ LT.lt ((Ordinal.typein (Prod.Lex s r)).toRelEmbedding { fst := b, snd := a } …
  -/
  have := ((add_lt_add_iff_left _).2 (typein_lt_type _ a)).trans_le this
  /-
    α β : Type u_4
    r : α → α → Prop
    s : β → β → Prop
    inst✝¹ : IsWellOrder α r
    inst✝ : IsWellOrder β s
    c : Ordinal.{u_4}
    h : (Ordinal.type s).IsLimit
    H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
    l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
    a : α
    b : β
    this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
    this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
    ⊢ LT.lt ((Ordinal.typein (Prod.Lex s r)).toRelEmbedding { fst := b, snd := a } …
  -/
  refine (RelEmbedding.ofMonotone (fun a => ?_) fun a b => ?_).ordinal_type_le.trans_lt this
    /-
      case refine_1
      α β : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      c : Ordinal.{u_4}
      h : (Ordinal.type s).IsLimit
      H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
      l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
      a✝ : α
      b : β
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
      this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
      a : Subtype fun b_1 => Prod.Lex s r b_1 { fst := b, snd := a✝ }
      ⊢ Sum (Prod (Subtype fun b_1 => s b_1 b) α) (Subtype fun b => r b a✝)
    -/
  · rcases a with ⟨⟨b', a'⟩, h⟩
    /-
      case refine_1.mk.mk
      α β : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      c : Ordinal.{u_4}
      h✝ : (Ordinal.type s).IsLimit
      H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
      l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
      a : α
      b : β
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
      this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
      b' : β
      a' : α
      h : Prod.Lex s r { fst := b', snd := a' } { fst := b, snd := a }
      ⊢ Sum (Prod (Subtype fun b_1 => s b_1 b) α) (Subtype fun b => r b a)
    -/
    by_cases e : b = b'
      /-
        case pos
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b' : β
        a' : α
        h : Prod.Lex s r { fst := b', snd := a' } { fst := b, snd := a }
        e : Eq b b'
        ⊢ Sum (Prod (Subtype fun b_1 => s b_1 b) α) (Subtype fun b => r b a)
      -/
    · refine Sum.inr ⟨a', ?_⟩
      /-
        case pos
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b' : β
        a' : α
        h : Prod.Lex s r { fst := b', snd := a' } { fst := b, snd := a }
        e : Eq b b'
        ⊢ r a' a
      -/
      subst e
      /-
        case pos
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        a' : α
        h : Prod.Lex s r { fst := b, snd := a' } { fst := b, snd := a }
        ⊢ r a' a
      -/
      cases' h with _ _ _ _ h _ _ _ h
        /-
          case pos.left
          α β : Type u_4
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          c : Ordinal.{u_4}
          h✝ : (Ordinal.type s).IsLimit
          H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
          l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
          a : α
          b : β
          this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
          this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
          a' : α
          h : s b b
          ⊢ r a' a
        -/
      · exact (irrefl _ h).elim
        /-
          🎉 no goals
        -/
        /-
          case pos.right
          α β : Type u_4
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          c : Ordinal.{u_4}
          h✝ : (Ordinal.type s).IsLimit
          H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
          l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
          a : α
          b : β
          this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
          this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
          a' : α
          h : r a' a
          ⊢ r a' a
        -/
      · exact h
        /-
          🎉 no goals
        -/
      /-
        case neg
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b' : β
        a' : α
        h : Prod.Lex s r { fst := b', snd := a' } { fst := b, snd := a }
        e : Not (Eq b b')
        ⊢ Sum (Prod (Subtype fun b_1 => s b_1 b) α) (Subtype fun b => r b a)
      -/
    · refine Sum.inl (⟨b', ?_⟩, a')
      /-
        case neg
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b' : β
        a' : α
        h : Prod.Lex s r { fst := b', snd := a' } { fst := b, snd := a }
        e : Not (Eq b b')
        ⊢ s b' b
      -/
      cases' h with _ _ _ _ h _ _ _ h
        /-
          case neg.left
          α β : Type u_4
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          c : Ordinal.{u_4}
          h✝ : (Ordinal.type s).IsLimit
          H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
          l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
          a : α
          b : β
          this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
          this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
          b' : β
          a' : α
          e : Not (Eq b b')
          h : s b' b
          ⊢ s b' b
        -/
      · exact h
        /-
          🎉 no goals
        -/
        /-
          case neg.right
          α β : Type u_4
          r : α → α → Prop
          s : β → β → Prop
          inst✝¹ : IsWellOrder α r
          inst✝ : IsWellOrder β s
          c : Ordinal.{u_4}
          h✝ : (Ordinal.type s).IsLimit
          H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
          l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
          a : α
          b : β
          this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
          this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
          a' : α
          e : Not (Eq b b)
          h : r a' a
          ⊢ s b b
        -/
      · exact (e rfl).elim
        /-
          🎉 no goals
        -/
    /-
      case refine_2
      α β : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      c : Ordinal.{u_4}
      h : (Ordinal.type s).IsLimit
      H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
      l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
      a✝ : α
      b✝ : β
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
      this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
      a b : Subtype fun b => Prod.Lex s r b { fst := b✝, snd := a✝ }
      ⊢ Subrel (Prod.Lex s r) (setOf fun b => Prod.Lex s r b { fst := b✝, snd := a✝  …
    -/
  · rcases a with ⟨⟨b₁, a₁⟩, h₁⟩
    /-
      case refine_2.mk.mk
      α β : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      c : Ordinal.{u_4}
      h : (Ordinal.type s).IsLimit
      H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
      l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
      a : α
      b✝ : β
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
      this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
      b : Subtype fun b => Prod.Lex s r b { fst := b✝, snd := a }
      b₁ : β
      a₁ : α
      h₁ : Prod.Lex s r { fst := b₁, snd := a₁ } { fst := b✝, snd := a }
      ⊢ Subrel (Prod.Lex s r) (setOf fun b => Prod.Lex s r b { fst := b✝, snd := a } …
    -/
    rcases b with ⟨⟨b₂, a₂⟩, h₂⟩
    /-
      case refine_2.mk.mk.mk.mk
      α β : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      c : Ordinal.{u_4}
      h : (Ordinal.type s).IsLimit
      H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
      l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
      a : α
      b : β
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
      this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
      b₁ : β
      a₁ : α
      h₁ : Prod.Lex s r { fst := b₁, snd := a₁ } { fst := b, snd := a }
      b₂ : β
      a₂ : α
      h₂ : Prod.Lex s r { fst := b₂, snd := a₂ } { fst := b, snd := a }
      ⊢ Subrel (Prod.Lex s r) (setOf fun b_1 => Prod.Lex s r b_1 { fst := b, snd :=  …
    -/
    intro h
    /-
      case refine_2.mk.mk.mk.mk
      α β : Type u_4
      r : α → α → Prop
      s : β → β → Prop
      inst✝¹ : IsWellOrder α r
      inst✝ : IsWellOrder β s
      c : Ordinal.{u_4}
      h✝ : (Ordinal.type s).IsLimit
      H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
      l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
      a : α
      b : β
      this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
      this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
      b₁ : β
      a₁ : α
      h₁ : Prod.Lex s r { fst := b₁, snd := a₁ } { fst := b, snd := a }
      b₂ : β
      a₂ : α
      h₂ : Prod.Lex s r { fst := b₂, snd := a₂ } { fst := b, snd := a }
      h : Subrel (Prod.Lex s r) (setOf fun b_1 => Prod.Lex s r b_1 { fst := b, snd : …
      ⊢ Sum.Lex (Prod.Lex (Subrel s (setOf fun b_1 => s b_1 b)) r) (Subrel r (setOf  …
    -/
    by_cases e₁ : b = b₁ <;> by_cases e₂ : b = b₂
      /-
        case pos
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b₁ : β
        a₁ : α
        h₁ : Prod.Lex s r { fst := b₁, snd := a₁ } { fst := b, snd := a }
        b₂ : β
        a₂ : α
        h₂ : Prod.Lex s r { fst := b₂, snd := a₂ } { fst := b, snd := a }
        h : Subrel (Prod.Lex s r) (setOf fun b_1 => Prod.Lex s r b_1 { fst := b, snd : …
        e₁ : Eq b b₁
        e₂ : Eq b b₂
        ⊢ Sum.Lex (Prod.Lex (Subrel s (setOf fun b_1 => s b_1 b)) r) (Subrel r (setOf  …
      -/
    · substs b₁ b₂
      simpa only [subrel_val, Prod.lex_def, @irrefl _ s _ b, true_and, false_or,
        eq_self_iff_true, dif_pos, Sum.lex_inr_inr] using h
      /-
        case neg
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b₁ : β
        a₁ : α
        h₁ : Prod.Lex s r { fst := b₁, snd := a₁ } { fst := b, snd := a }
        b₂ : β
        a₂ : α
        h₂ : Prod.Lex s r { fst := b₂, snd := a₂ } { fst := b, snd := a }
        h : Subrel (Prod.Lex s r) (setOf fun b_1 => Prod.Lex s r b_1 { fst := b, snd : …
        e₁ : Eq b b₁
        e₂ : Not (Eq b b₂)
        ⊢ Sum.Lex (Prod.Lex (Subrel s (setOf fun b_1 => s b_1 b)) r) (Subrel r (setOf  …
      -/
    · subst b₁
      simp only [subrel_val, Prod.lex_def, e₂, Prod.lex_def, dif_pos, subrel_val, eq_self_iff_true,
        or_false, dif_neg, not_false_iff, Sum.lex_inr_inl, false_and] at h ⊢
      /-
        case neg
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        a₁ : α
        b₂ : β
        a₂ : α
        h₂ : Prod.Lex s r { fst := b₂, snd := a₂ } { fst := b, snd := a }
        e₂ : Not (Eq b b₂)
        h₁ : Prod.Lex s r { fst := b, snd := a₁ } { fst := b, snd := a }
        h : s b b₂
        ⊢ False
      -/
      cases' h₂ with _ _ _ _ h₂_h h₂_h <;> [exact asymm h h₂_h; exact e₂ rfl]
      /-
        🎉 no goals
      -/
      /-
        case pos
        α β : Type u_4
        r : α → α → Prop
        s : β → β → Prop
        inst✝¹ : IsWellOrder α r
        inst✝ : IsWellOrder β s
        c : Ordinal.{u_4}
        h✝ : (Ordinal.type s).IsLimit
        H : ∀ (b' : Ordinal.{u_4}), LT.lt b' (Ordinal.type s) → LE.le (HMul.hMul (Ordi …
        l : LT.lt c (HMul.hMul (Ordinal.type r) (Ordinal.type s))
        a : α
        b : β
        this✝ : LE.le (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRel …
        this : LT.lt (HAdd.hAdd (HMul.hMul (Ordinal.type r) ((Ordinal.typein s).toRelE …
        b₁ : β
        a₁ : α
        h₁ : Prod.Lex s r { fst := b₁, snd := a₁ } { fst := b, snd := a }
        b₂ : β
        a₂ : α
        h₂ : Prod.Lex s r { fst := b₂, snd := a₂ } { fst := b, snd := a }
        h : Subrel (Prod.Lex s r) (setOf fun b_1 => Prod.Lex s r b_1 { fst := b, snd : …
        e₁ : Not (Eq b b₁)
        e₂ : Eq b b₂
        ⊢ Sum.Lex (Prod.Lex (Subrel s (setOf fun b_1 => s b_1 b)) r) (Subrel r (setOf  …
      -/
    · simp [e₂, dif_neg e₁, show b₂ ≠ b₁ from e₂ ▸ e₁]
      /-
        🎉 no goals
      -/
    · simpa only [dif_neg e₁, dif_neg e₂, Prod.lex_def, subrel_val, Subtype.mk_eq_mk,
        Sum.lex_inl_inl] using h


theorem mul_le_of_limit {a b c : Ordinal} (h : IsLimit b) : a * b ≤ c ↔ ∀ b' < b, a * b' ≤ c :=
  ⟨fun h _ l => (mul_le_mul_left' l.le _).trans h, fun H =>
    -- Porting note: `induction` tactics are required because of the parser bug.
    le_of_not_lt <| by
      induction a using inductionOn with
      | H α r =>
        induction b using inductionOn with
        | H β s =>
          exact mul_le_of_limit_aux h H⟩


theorem isNormal_mul_right {a : Ordinal} (h : 0 < a) : IsNormal (a * ·) :=
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
  ⟨fun b => by
      /-
        a : Ordinal.{u_4}
        h : LT.lt 0 a
        b : Ordinal.{u_4}
        ⊢ LT.lt ((fun x => HMul.hMul a x) b) ((fun x => HMul.hMul a x) (Order.succ b))
      -/
      beta_reduce
      /-
        a : Ordinal.{u_4}
        h : LT.lt 0 a
        b : Ordinal.{u_4}
        ⊢ LT.lt (HMul.hMul a b) (HMul.hMul a (Order.succ b))
      -/
      rw [mul_succ]
      /-
        a : Ordinal.{u_4}
        h : LT.lt 0 a
        b : Ordinal.{u_4}
        ⊢ LT.lt (HMul.hMul a b) (HAdd.hAdd (HMul.hMul a b) a)
      -/
      simpa only [add_zero] using (add_lt_add_iff_left (a * b)).2 h,
      /-
        🎉 no goals
      -/
    fun _ l _ => mul_le_of_limit l⟩


@[deprecated isNormal_mul_right (since := "2024-10-11")]
alias mul_isNormal := isNormal_mul_right


theorem lt_mul_of_limit {a b c : Ordinal} (h : IsLimit c) : a < b * c ↔ ∃ c' < c, a < b * c' := by
  -- Porting note: `bex_def` is required.
  /-
    a b c : Ordinal.{u_4}
    h : c.IsLimit
    ⊢ Iff (LT.lt a (HMul.hMul b c)) (Exists fun c' => And (LT.lt c' c) (LT.lt a (H …
  -/
  simpa only [not_forall₂, not_le, bex_def] using not_congr (@mul_le_of_limit b c a h)
  /-
    🎉 no goals
  -/


theorem mul_lt_mul_iff_left {a b c : Ordinal} (a0 : 0 < a) : a * b < a * c ↔ b < c :=
  (isNormal_mul_right a0).lt_iff


theorem mul_le_mul_iff_left {a b c : Ordinal} (a0 : 0 < a) : a * b ≤ a * c ↔ b ≤ c :=
  (isNormal_mul_right a0).le_iff


theorem mul_lt_mul_of_pos_left {a b c : Ordinal} (h : a < b) (c0 : 0 < c) : c * a < c * b :=
  (mul_lt_mul_iff_left c0).2 h


theorem mul_pos {a b : Ordinal} (h₁ : 0 < a) (h₂ : 0 < b) : 0 < a * b := by
  /-
    a b : Ordinal.{u_4}
    h₁ : LT.lt 0 a
    h₂ : LT.lt 0 b
    ⊢ LT.lt 0 (HMul.hMul a b)
  -/
  simpa only [mul_zero] using mul_lt_mul_of_pos_left h₂ h₁
  /-
    🎉 no goals
  -/


theorem mul_ne_zero {a b : Ordinal} : a ≠ 0 → b ≠ 0 → a * b ≠ 0 := by
  /-
    a b : Ordinal.{u_4}
    ⊢ Ne a 0 → Ne b 0 → Ne (HMul.hMul a b) 0
  -/
  simpa only [Ordinal.pos_iff_ne_zero] using mul_pos
  /-
    🎉 no goals
  -/


theorem le_of_mul_le_mul_left {a b c : Ordinal} (h : c * a ≤ c * b) (h0 : 0 < c) : a ≤ b :=
  le_imp_le_of_lt_imp_lt (fun h' => mul_lt_mul_of_pos_left h' h0) h


theorem mul_right_inj {a b c : Ordinal} (a0 : 0 < a) : a * b = a * c ↔ b = c :=
  (isNormal_mul_right a0).inj


theorem isLimit_mul {a b : Ordinal} (a0 : 0 < a) : IsLimit b → IsLimit (a * b) :=
  (isNormal_mul_right a0).isLimit


@[deprecated isLimit_mul (since := "2024-10-11")]
alias mul_isLimit := isLimit_mul


theorem isLimit_mul_left {a b : Ordinal} (l : IsLimit a) (b0 : 0 < b) : IsLimit (a * b) := by
  /-
    a b : Ordinal.{u_4}
    l : a.IsLimit
    b0 : LT.lt 0 b
    ⊢ (HMul.hMul a b).IsLimit
  -/
  rcases zero_or_succ_or_limit b with (rfl | ⟨b, rfl⟩ | lb)
    /-
      case inl
      a : Ordinal.{u_4}
      l : a.IsLimit
      b0 : LT.lt 0 0
      ⊢ (HMul.hMul a 0).IsLimit
    -/
  · exact b0.false.elim
    /-
      🎉 no goals
    -/
    /-
      case inr.inl.intro
      a : Ordinal.{u_4}
      l : a.IsLimit
      b : Ordinal.{u_4}
      b0 : LT.lt 0 (Order.succ b)
      ⊢ (HMul.hMul a (Order.succ b)).IsLimit
    -/
  · rw [mul_succ]
    /-
      case inr.inl.intro
      a : Ordinal.{u_4}
      l : a.IsLimit
      b : Ordinal.{u_4}
      b0 : LT.lt 0 (Order.succ b)
      ⊢ (HAdd.hAdd (HMul.hMul a b) a).IsLimit
    -/
    exact isLimit_add _ l
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      a b : Ordinal.{u_4}
      l : a.IsLimit
      b0 : LT.lt 0 b
      lb : b.IsLimit
      ⊢ (HMul.hMul a b).IsLimit
    -/
  · exact isLimit_mul l.pos lb
    /-
      🎉 no goals
    -/


@[deprecated isLimit_mul_left (since := "2024-10-11")]
alias mul_isLimit_left := isLimit_mul_left


theorem smul_eq_mul : ∀ (n : ℕ) (a : Ordinal), n • a = a * n
               /-
                 a : Ordinal.{u_4}
                 ⊢ Eq (HSMul.hSMul 0 a) (HMul.hMul a ↑0)
               -/
  | 0, a => by rw [zero_nsmul, Nat.cast_zero, mul_zero]
               /-
                 🎉 no goals
               -/
                   /-
                     n : Nat
                     a : Ordinal.{u_4}
                     ⊢ Eq (HSMul.hSMul (HAdd.hAdd n 1) a) (HMul.hMul a ↑(HAdd.hAdd n 1))
                   -/
  | n + 1, a => by rw [succ_nsmul, Nat.cast_add, mul_add, Nat.cast_one, mul_one, smul_eq_mul n]
                   /-
                     🎉 no goals
                   -/


private theorem add_mul_limit_aux {a b c : Ordinal} (ba : b + a = a) (l : IsLimit c)
    (IH : ∀ c' < c, (a + b) * succ c' = a * succ c' + b) : (a + b) * c = a * c :=
  le_antisymm
    ((mul_le_of_limit l).2 fun c' h => by
      /-
        a b c : Ordinal.{u_4}
        ba : Eq (HAdd.hAdd b a) a
        l : c.IsLimit
        IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
        c' : Ordinal.{u_4}
        h : LT.lt c' c
        ⊢ LE.le (HMul.hMul (HAdd.hAdd a b) c') (HMul.hMul a c)
      -/
      apply (mul_le_mul_left' (le_succ c') _).trans
      /-
        a b c : Ordinal.{u_4}
        ba : Eq (HAdd.hAdd b a) a
        l : c.IsLimit
        IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
        c' : Ordinal.{u_4}
        h : LT.lt c' c
        ⊢ LE.le (HMul.hMul (HAdd.hAdd a b) (Order.succ c')) (HMul.hMul a c)
      -/
      rw [IH _ h]
      /-
        a b c : Ordinal.{u_4}
        ba : Eq (HAdd.hAdd b a) a
        l : c.IsLimit
        IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
        c' : Ordinal.{u_4}
        h : LT.lt c' c
        ⊢ LE.le (HAdd.hAdd (HMul.hMul a (Order.succ c')) b) (HMul.hMul a c)
      -/
      apply (add_le_add_left _ _).trans
        /-
          a b c : Ordinal.{u_4}
          ba : Eq (HAdd.hAdd b a) a
          l : c.IsLimit
          IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
          c' : Ordinal.{u_4}
          h : LT.lt c' c
          ⊢ LE.le (HAdd.hAdd (HMul.hMul a (Order.succ c')) ?m.123365) (HMul.hMul a c)
        -/
      · rw [← mul_succ]
        /-
          a b c : Ordinal.{u_4}
          ba : Eq (HAdd.hAdd b a) a
          l : c.IsLimit
          IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
          c' : Ordinal.{u_4}
          h : LT.lt c' c
          ⊢ LE.le (HMul.hMul a (Order.succ (Order.succ c'))) (HMul.hMul a c)
        -/
        exact mul_le_mul_left' (succ_le_of_lt <| l.succ_lt h) _
        /-
          🎉 no goals
        -/
        /-
          a b c : Ordinal.{u_4}
          ba : Eq (HAdd.hAdd b a) a
          l : c.IsLimit
          IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
          c' : Ordinal.{u_4}
          h : LT.lt c' c
          ⊢ LE.le b a
        -/
      · rw [← ba]
        /-
          a b c : Ordinal.{u_4}
          ba : Eq (HAdd.hAdd b a) a
          l : c.IsLimit
          IH : ∀ (c' : Ordinal.{u_4}), LT.lt c' c → Eq (HMul.hMul (HAdd.hAdd a b) (Order …
          c' : Ordinal.{u_4}
          h : LT.lt c' c
          ⊢ LE.le b (HAdd.hAdd b a)
        -/
        exact le_add_right _ _)
        /-
          🎉 no goals
        -/
    (mul_le_mul_right' (le_add_right _ _) _)


theorem add_mul_succ {a b : Ordinal} (c) (ba : b + a = a) : (a + b) * succ c = a * succ c + b := by
  induction c using limitRecOn with
  | H₁ => simp only [succ_zero, mul_one]
  | H₂ c IH =>
    rw [mul_succ, IH, ← add_assoc, add_assoc _ b, ba, ← mul_succ]
  | H₃ c l IH =>
    rw [mul_succ, add_mul_limit_aux ba l IH, mul_succ, add_assoc]


theorem add_mul_limit {a b c : Ordinal} (ba : b + a = a) (l : IsLimit c) : (a + b) * c = a * c :=
  add_mul_limit_aux ba l fun c' _ => add_mul_succ c' ba


/-- The set in the definition of division is nonempty. -/
private theorem div_nonempty {a b : Ordinal} (h : b ≠ 0) : { o | a < b * succ o }.Nonempty :=
  ⟨a, (succ_le_iff (a := a) (b := b * succ a)).1 <| by
    simpa only [succ_zero, one_mul] using
      mul_le_mul_right' (succ_le_of_lt (Ordinal.pos_iff_ne_zero.2 h)) (succ a)⟩


/-- `a / b` is the unique ordinal `o` satisfying `a = b * o + o'` with `o' < b`. -/
instance div : Div Ordinal :=
  ⟨fun a b => if b = 0 then 0 else sInf { o | a < b * succ o }⟩


@[simp]
theorem div_zero (a : Ordinal) : a / 0 = 0 :=
  dif_pos rfl


private theorem div_def (a) {b : Ordinal} (h : b ≠ 0) : a / b = sInf { o | a < b * succ o } :=
  dif_neg h


theorem lt_mul_succ_div (a) {b : Ordinal} (h : b ≠ 0) : a < b * succ (a / b) := by
  /-
    a b : Ordinal.{u_4}
    h : Ne b 0
    ⊢ LT.lt a (HMul.hMul b (Order.succ (HDiv.hDiv a b)))
  -/
  rw [div_def a h]; exact csInf_mem (div_nonempty h)
                    /-
                      🎉 no goals
                    -/


theorem lt_mul_div_add (a) {b : Ordinal} (h : b ≠ 0) : a < b * (a / b) + b := by
  /-
    a b : Ordinal.{u_4}
    h : Ne b 0
    ⊢ LT.lt a (HAdd.hAdd (HMul.hMul b (HDiv.hDiv a b)) b)
  -/
  simpa only [mul_succ] using lt_mul_succ_div a h
  /-
    🎉 no goals
  -/


theorem div_le {a b c : Ordinal} (b0 : b ≠ 0) : a / b ≤ c ↔ a < b * succ c :=
  ⟨fun h => (lt_mul_succ_div a b0).trans_le (mul_le_mul_left' (succ_le_succ_iff.2 h) _), fun h => by
    /-
      a b c : Ordinal.{u_4}
      b0 : Ne b 0
      h : LT.lt a (HMul.hMul b (Order.succ c))
      ⊢ LE.le (HDiv.hDiv a b) c
    -/
    rw [div_def a b0]; exact csInf_le' h⟩
                       /-
                         🎉 no goals
                       -/


theorem lt_div {a b c : Ordinal} (h : c ≠ 0) : a < b / c ↔ c * succ a ≤ b := by
  /-
    a b c : Ordinal.{u_4}
    h : Ne c 0
    ⊢ Iff (LT.lt a (HDiv.hDiv b c)) (LE.le (HMul.hMul c (Order.succ a)) b)
  -/
  rw [← not_le, div_le h, not_lt]
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        b c : Ordinal.{u_4}
                                                                        h : Ne c 0
                                                                        ⊢ Iff (LT.lt 0 (HDiv.hDiv b c)) (LE.le c b)
                                                                      -/
theorem div_pos {b c : Ordinal} (h : c ≠ 0) : 0 < b / c ↔ c ≤ b := by simp [lt_div h]
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem le_div {a b c : Ordinal} (c0 : c ≠ 0) : a ≤ b / c ↔ c * a ≤ b := by
  induction a using limitRecOn with
  | H₁ => simp only [mul_zero, Ordinal.zero_le]
  | H₂ _ _ => rw [succ_le_iff, lt_div c0]
  | H₃ _ h₁ h₂ =>
    revert h₁ h₂
    simp +contextual only [mul_le_of_limit, limit_le, forall_true_iff]


theorem div_lt {a b c : Ordinal} (b0 : b ≠ 0) : a / b < c ↔ a < b * c :=
  lt_iff_lt_of_le_iff_le <| le_div b0


theorem div_le_of_le_mul {a b c : Ordinal} (h : a ≤ b * c) : a / b ≤ c :=
                        /-
                          a b c : Ordinal.{u_4}
                          h : LE.le a (HMul.hMul b c)
                          b0 : Eq b 0
                          ⊢ LE.le (HDiv.hDiv a b) c
                        -/
  if b0 : b = 0 then by simp only [b0, div_zero, Ordinal.zero_le]
                        /-
                          🎉 no goals
                        -/
  else
    (div_le b0).2 <| h.trans_lt <| mul_lt_mul_of_pos_left (lt_succ c) (Ordinal.pos_iff_ne_zero.2 b0)


theorem mul_lt_of_lt_div {a b c : Ordinal} : a < b / c → c * a < b :=
  lt_imp_lt_of_le_imp_le div_le_of_le_mul


@[simp]
theorem zero_div (a : Ordinal) : 0 / a = 0 :=
  Ordinal.le_zero.1 <| div_le_of_le_mul <| Ordinal.zero_le _


theorem mul_div_le (a b : Ordinal) : b * (a / b) ≤ a :=
                        /-
                          a b : Ordinal.{u_4}
                          b0 : Eq b 0
                          ⊢ LE.le (HMul.hMul b (HDiv.hDiv a b)) a
                        -/
  if b0 : b = 0 then by simp only [b0, zero_mul, Ordinal.zero_le] else (le_div b0).1 le_rfl
                        /-
                          🎉 no goals
                        -/


theorem div_le_left {a b : Ordinal} (h : a ≤ b) (c : Ordinal) : a / c ≤ b / c := by
  /-
    a b : Ordinal.{u_4}
    h : LE.le a b
    c : Ordinal.{u_4}
    ⊢ LE.le (HDiv.hDiv a c) (HDiv.hDiv b c)
  -/
  obtain rfl | hc := eq_or_ne c 0
    /-
      case inl
      a b : Ordinal.{u_4}
      h : LE.le a b
      ⊢ LE.le (HDiv.hDiv a 0) (HDiv.hDiv b 0)
    -/
  · rw [div_zero, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a b : Ordinal.{u_4}
      h : LE.le a b
      c : Ordinal.{u_4}
      hc : Ne c 0
      ⊢ LE.le (HDiv.hDiv a c) (HDiv.hDiv b c)
    -/
  · rw [le_div hc]
    /-
      case inr
      a b : Ordinal.{u_4}
      h : LE.le a b
      c : Ordinal.{u_4}
      hc : Ne c 0
      ⊢ LE.le (HMul.hMul c (HDiv.hDiv a c)) b
    -/
    exact (mul_div_le a c).trans h
    /-
      🎉 no goals
    -/


theorem mul_add_div (a) {b : Ordinal} (b0 : b ≠ 0) (c) : (b * a + c) / b = a + c / b := by
  /-
    a b : Ordinal.{u_4}
    b0 : Ne b 0
    c : Ordinal.{u_4}
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul b a) c) b) (HAdd.hAdd a (HDiv.hDiv c b))
  -/
  apply le_antisymm
    /-
      case a
      a b : Ordinal.{u_4}
      b0 : Ne b 0
      c : Ordinal.{u_4}
      ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HMul.hMul b a) c) b) (HAdd.hAdd a (HDiv.hDiv c  …
    -/
  · apply (div_le b0).2
    /-
      case a
      a b : Ordinal.{u_4}
      b0 : Ne b 0
      c : Ordinal.{u_4}
      ⊢ LT.lt (HAdd.hAdd (HMul.hMul b a) c) (HMul.hMul b (Order.succ (HAdd.hAdd a (H …
    -/
    rw [mul_succ, mul_add, add_assoc, add_lt_add_iff_left]
    /-
      case a
      a b : Ordinal.{u_4}
      b0 : Ne b 0
      c : Ordinal.{u_4}
      ⊢ LT.lt c (HAdd.hAdd (HMul.hMul b (HDiv.hDiv c b)) b)
    -/
    apply lt_mul_div_add _ b0
    /-
      🎉 no goals
    -/
    /-
      case a
      a b : Ordinal.{u_4}
      b0 : Ne b 0
      c : Ordinal.{u_4}
      ⊢ LE.le (HAdd.hAdd a (HDiv.hDiv c b)) (HDiv.hDiv (HAdd.hAdd (HMul.hMul b a) c) …
    -/
  · rw [le_div b0, mul_add, add_le_add_iff_left]
    /-
      case a
      a b : Ordinal.{u_4}
      b0 : Ne b 0
      c : Ordinal.{u_4}
      ⊢ LE.le (HMul.hMul b (HDiv.hDiv c b)) c
    -/
    apply mul_div_le
    /-
      🎉 no goals
    -/


theorem div_eq_zero_of_lt {a b : Ordinal} (h : a < b) : a / b = 0 := by
  /-
    a b : Ordinal.{u_4}
    h : LT.lt a b
    ⊢ Eq (HDiv.hDiv a b) 0
  -/
  rw [← Ordinal.le_zero, div_le <| Ordinal.pos_iff_ne_zero.1 <| (Ordinal.zero_le _).trans_lt h]
  /-
    a b : Ordinal.{u_4}
    h : LT.lt a b
    ⊢ LT.lt a (HMul.hMul b (Order.succ 0))
  -/
  simpa only [succ_zero, mul_one] using h
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_div_cancel (a) {b : Ordinal} (b0 : b ≠ 0) : b * a / b = a := by
  /-
    a b : Ordinal.{u_4}
    b0 : Ne b 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul b a) b) a
  -/
  simpa only [add_zero, zero_div] using mul_add_div a b0 0
  /-
    🎉 no goals
  -/


theorem mul_add_div_mul {a c : Ordinal} (hc : c < a) (b d : Ordinal) :
    (a * b + c) / (a * d) = b / d := by
  /-
    a c : Ordinal.{u_4}
    hc : LT.lt c a
    b d : Ordinal.{u_4}
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a d)) (HDiv.hDiv b d)
  -/
  have ha : a ≠ 0 := ((Ordinal.zero_le c).trans_lt hc).ne'
  /-
    a c : Ordinal.{u_4}
    hc : LT.lt c a
    b d : Ordinal.{u_4}
    ha : Ne a 0
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a d)) (HDiv.hDiv b d)
  -/
  obtain rfl | hd := eq_or_ne d 0
    /-
      case inl
      a c : Ordinal.{u_4}
      hc : LT.lt c a
      b : Ordinal.{u_4}
      ha : Ne a 0
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a 0)) (HDiv.hDiv b 0)
    -/
  · rw [mul_zero, div_zero, div_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      a c : Ordinal.{u_4}
      hc : LT.lt c a
      b d : Ordinal.{u_4}
      ha : Ne a 0
      hd : Ne d 0
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a d)) (HDiv.hDiv b d)
    -/
  · have H := mul_ne_zero ha hd
    /-
      case inr
      a c : Ordinal.{u_4}
      hc : LT.lt c a
      b d : Ordinal.{u_4}
      ha : Ne a 0
      hd : Ne d 0
      H : Ne (HMul.hMul a d) 0
      ⊢ Eq (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a d)) (HDiv.hDiv b d)
    -/
    apply le_antisymm
      /-
        case inr.a
        a c : Ordinal.{u_4}
        hc : LT.lt c a
        b d : Ordinal.{u_4}
        ha : Ne a 0
        hd : Ne d 0
        H : Ne (HMul.hMul a d) 0
        ⊢ LE.le (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a d)) (HDiv.hDiv b …
      -/
    · rw [← lt_succ_iff, div_lt H, mul_assoc]
        /-
          case inr.a
          a c : Ordinal.{u_4}
          hc : LT.lt c a
          b d : Ordinal.{u_4}
          ha : Ne a 0
          hd : Ne d 0
          H : Ne (HMul.hMul a d) 0
          ⊢ LT.lt (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a (HMul.hMul d (Order.succ (H …
        -/
      · apply (add_lt_add_left hc _).trans_le
        /-
          case inr.a
          a c : Ordinal.{u_4}
          hc : LT.lt c a
          b d : Ordinal.{u_4}
          ha : Ne a 0
          hd : Ne d 0
          H : Ne (HMul.hMul a d) 0
          ⊢ LE.le (HAdd.hAdd (HMul.hMul a b) a) (HMul.hMul a (HMul.hMul d (Order.succ (H …
        -/
        rw [← mul_succ]
        /-
          case inr.a
          a c : Ordinal.{u_4}
          hc : LT.lt c a
          b d : Ordinal.{u_4}
          ha : Ne a 0
          hd : Ne d 0
          H : Ne (HMul.hMul a d) 0
          ⊢ LE.le (HMul.hMul a (Order.succ b)) (HMul.hMul a (HMul.hMul d (Order.succ (HD …
        -/
        apply mul_le_mul_left'
        /-
          case inr.a.bc
          a c : Ordinal.{u_4}
          hc : LT.lt c a
          b d : Ordinal.{u_4}
          ha : Ne a 0
          hd : Ne d 0
          H : Ne (HMul.hMul a d) 0
          ⊢ LE.le (Order.succ b) (HMul.hMul d (Order.succ (HDiv.hDiv b d)))
        -/
        rw [succ_le_iff]
        /-
          case inr.a.bc
          a c : Ordinal.{u_4}
          hc : LT.lt c a
          b d : Ordinal.{u_4}
          ha : Ne a 0
          hd : Ne d 0
          H : Ne (HMul.hMul a d) 0
          ⊢ LT.lt b (HMul.hMul d (Order.succ (HDiv.hDiv b d)))
        -/
        exact lt_mul_succ_div b hd
        /-
          🎉 no goals
        -/
      /-
        case inr.a
        a c : Ordinal.{u_4}
        hc : LT.lt c a
        b d : Ordinal.{u_4}
        ha : Ne a 0
        hd : Ne d 0
        H : Ne (HMul.hMul a d) 0
        ⊢ LE.le (HDiv.hDiv b d) (HDiv.hDiv (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a  …
      -/
    · rw [le_div H, mul_assoc]
      /-
        case inr.a
        a c : Ordinal.{u_4}
        hc : LT.lt c a
        b d : Ordinal.{u_4}
        ha : Ne a 0
        hd : Ne d 0
        H : Ne (HMul.hMul a d) 0
        ⊢ LE.le (HMul.hMul a (HMul.hMul d (HDiv.hDiv b d))) (HAdd.hAdd (HMul.hMul a b) …
      -/
      exact (mul_le_mul_left' (mul_div_le b d) a).trans (le_add_right _ c)
      /-
        🎉 no goals
      -/


theorem mul_div_mul_cancel {a : Ordinal} (ha : a ≠ 0) (b c) : a * b / (a * c) = b / c := by
  /-
    a : Ordinal.{u_4}
    ha : Ne a 0
    b c : Ordinal.{u_4}
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul a c)) (HDiv.hDiv b c)
  -/
  convert mul_add_div_mul (Ordinal.pos_iff_ne_zero.2 ha) b c using 1
  /-
    case h.e'_2
    a : Ordinal.{u_4}
    ha : Ne a 0
    b c : Ordinal.{u_4}
    ⊢ Eq (HDiv.hDiv (HMul.hMul a b) (HMul.hMul a c)) (HDiv.hDiv (HAdd.hAdd (HMul.h …
  -/
  rw [add_zero]
  /-
    🎉 no goals
  -/


@[simp]
theorem div_one (a : Ordinal) : a / 1 = a := by
  /-
    a : Ordinal.{u_4}
    ⊢ Eq (HDiv.hDiv a 1) a
  -/
  simpa only [one_mul] using mul_div_cancel a Ordinal.one_ne_zero
  /-
    🎉 no goals
  -/


@[simp]
theorem div_self {a : Ordinal} (h : a ≠ 0) : a / a = 1 := by
  /-
    a : Ordinal.{u_4}
    h : Ne a 0
    ⊢ Eq (HDiv.hDiv a a) 1
  -/
  simpa only [mul_one] using mul_div_cancel 1 h
  /-
    🎉 no goals
  -/


theorem mul_sub (a b c : Ordinal) : a * (b - c) = a * b - a * c :=
                        /-
                          a b c : Ordinal.{u_4}
                          a0 : Eq a 0
                          ⊢ Eq (HMul.hMul a (HSub.hSub b c)) (HSub.hSub (HMul.hMul a b) (HMul.hMul a c))
                        -/
  if a0 : a = 0 then by simp only [a0, zero_mul, sub_self]
                        /-
                          🎉 no goals
                        -/
  else
                                    /-
                                      a b c : Ordinal.{u_4}
                                      a0 : Not (Eq a 0)
                                      d : Ordinal.{u_4}
                                      ⊢ Iff (LE.le (HMul.hMul a (HSub.hSub b c)) d) (LE.le (HSub.hSub (HMul.hMul a b …
                                    -/
    eq_of_forall_ge_iff fun d => by rw [sub_le, ← le_div a0, sub_le, ← le_div a0, mul_add_div _ a0]
                                    /-
                                      🎉 no goals
                                    -/


theorem isLimit_add_iff {a b} : IsLimit (a + b) ↔ IsLimit b ∨ b = 0 ∧ IsLimit a := by
  /-
    a b : Ordinal.{u_4}
    ⊢ Iff (HAdd.hAdd a b).IsLimit (Or b.IsLimit (And (Eq b 0) a.IsLimit))
  -/
  constructor <;> intro h
    /-
      case mp
      a b : Ordinal.{u_4}
      h : (HAdd.hAdd a b).IsLimit
      ⊢ Or b.IsLimit (And (Eq b 0) a.IsLimit)
    -/
  · by_cases h' : b = 0
      /-
        case pos
        a b : Ordinal.{u_4}
        h : (HAdd.hAdd a b).IsLimit
        h' : Eq b 0
        ⊢ Or b.IsLimit (And (Eq b 0) a.IsLimit)
      -/
    · rw [h', add_zero] at h
      /-
        case pos
        a b : Ordinal.{u_4}
        h : a.IsLimit
        h' : Eq b 0
        ⊢ Or b.IsLimit (And (Eq b 0) a.IsLimit)
      -/
      right
      /-
        case pos.h
        a b : Ordinal.{u_4}
        h : a.IsLimit
        h' : Eq b 0
        ⊢ And (Eq b 0) a.IsLimit
      -/
      exact ⟨h', h⟩
      /-
        🎉 no goals
      -/
    /-
      case neg
      a b : Ordinal.{u_4}
      h : (HAdd.hAdd a b).IsLimit
      h' : Not (Eq b 0)
      ⊢ Or b.IsLimit (And (Eq b 0) a.IsLimit)
    -/
    left
    /-
      case neg.h
      a b : Ordinal.{u_4}
      h : (HAdd.hAdd a b).IsLimit
      h' : Not (Eq b 0)
      ⊢ b.IsLimit
    -/
    rw [← add_sub_cancel a b]
    /-
      case neg.h
      a b : Ordinal.{u_4}
      h : (HAdd.hAdd a b).IsLimit
      h' : Not (Eq b 0)
      ⊢ (HSub.hSub (HAdd.hAdd a b) a).IsLimit
    -/
    apply isLimit_sub h
    /-
      case neg.h
      a b : Ordinal.{u_4}
      h : (HAdd.hAdd a b).IsLimit
      h' : Not (Eq b 0)
      ⊢ LT.lt a (HAdd.hAdd a b)
    -/
    suffices a + 0 < a + b by simpa only [add_zero] using this
    /-
      case neg.h
      a b : Ordinal.{u_4}
      h : (HAdd.hAdd a b).IsLimit
      h' : Not (Eq b 0)
      ⊢ LT.lt (HAdd.hAdd a 0) (HAdd.hAdd a b)
    -/
    rwa [add_lt_add_iff_left, Ordinal.pos_iff_ne_zero]
    /-
      🎉 no goals
    -/
  /-
    case mpr
    a b : Ordinal.{u_4}
    h : Or b.IsLimit (And (Eq b 0) a.IsLimit)
    ⊢ (HAdd.hAdd a b).IsLimit
  -/
  rcases h with (h | ⟨rfl, h⟩)
    /-
      case mpr.inl
      a b : Ordinal.{u_4}
      h : b.IsLimit
      ⊢ (HAdd.hAdd a b).IsLimit
    -/
  · exact isLimit_add a h
    /-
      🎉 no goals
    -/
    /-
      case mpr.inr.intro
      a : Ordinal.{u_4}
      h : a.IsLimit
      ⊢ (HAdd.hAdd a 0).IsLimit
    -/
  · simpa only [add_zero]
    /-
      🎉 no goals
    -/


theorem dvd_add_iff : ∀ {a b c : Ordinal}, a ∣ b → (a ∣ b + c ↔ a ∣ c)
  | a, _, c, ⟨b, rfl⟩ =>
                              /-
                                a c b : Ordinal.{u_4}
                                x✝ : Dvd.dvd a (HAdd.hAdd (HMul.hMul a b) c)
                                d : Ordinal.{u_4}
                                e : Eq (HAdd.hAdd (HMul.hMul a b) c) (HMul.hMul a d)
                                ⊢ Eq c (HMul.hMul a (HSub.hSub d b))
                              -/
    ⟨fun ⟨d, e⟩ => ⟨d - b, by rw [mul_sub, ← e, add_sub_cancel]⟩, fun ⟨d, e⟩ => by
                              /-
                                🎉 no goals
                              -/
      /-
        a c b : Ordinal.{u_4}
        x✝ : Dvd.dvd a c
        d : Ordinal.{u_4}
        e : Eq c (HMul.hMul a d)
        ⊢ Dvd.dvd a (HAdd.hAdd (HMul.hMul a b) c)
      -/
      rw [e, ← mul_add]
      /-
        a c b : Ordinal.{u_4}
        x✝ : Dvd.dvd a c
        d : Ordinal.{u_4}
        e : Eq c (HMul.hMul a d)
        ⊢ Dvd.dvd a (HMul.hMul a (HAdd.hAdd b d))
      -/
      apply dvd_mul_right⟩
      /-
        🎉 no goals
      -/


theorem div_mul_cancel : ∀ {a b : Ordinal}, a ≠ 0 → a ∣ b → a * (b / a) = b
                             /-
                               a : Ordinal.{u_4}
                               a0 : Ne a 0
                               b : Ordinal.{u_4}
                               ⊢ Eq (HMul.hMul a (HDiv.hDiv (HMul.hMul a b) a)) (HMul.hMul a b)
                             -/
  | a, _, a0, ⟨b, rfl⟩ => by rw [mul_div_cancel _ a0]
                             /-
                               🎉 no goals
                             -/


theorem le_of_dvd : ∀ {a b : Ordinal}, b ≠ 0 → a ∣ b → a ≤ b
  -- Porting note: `⟨b, rfl⟩ => by` → `⟨b, e⟩ => by subst e`
  | a, _, b0, ⟨b, e⟩ => by
    /-
      a x✝ : Ordinal.{u_4}
      b0 : Ne x✝ 0
      b : Ordinal.{u_4}
      e : Eq x✝ (HMul.hMul a b)
      ⊢ LE.le a x✝
    -/
    subst e
    -- Porting note: `Ne` is required.
    simpa only [mul_one] using
      mul_le_mul_left'
        (one_le_iff_ne_zero.2 fun h : b = 0 => by
          simp only [h, mul_zero, Ne, not_true_eq_false] at b0) a


theorem dvd_antisymm {a b : Ordinal} (h₁ : a ∣ b) (h₂ : b ∣ a) : a = b :=
                        /-
                          a b : Ordinal.{u_4}
                          h₁ : Dvd.dvd a b
                          h₂ : Dvd.dvd b a
                          a0 : Eq a 0
                          ⊢ Eq a b
                        -/
  if a0 : a = 0 then by subst a; exact (eq_zero_of_zero_dvd h₁).symm
                                 /-
                                   🎉 no goals
                                 -/
  else
                          /-
                            a b : Ordinal.{u_4}
                            h₁ : Dvd.dvd a b
                            h₂ : Dvd.dvd b a
                            a0 : Not (Eq a 0)
                            b0 : Eq b 0
                            ⊢ Eq a b
                          -/
    if b0 : b = 0 then by subst b; exact eq_zero_of_zero_dvd h₂
                                   /-
                                     🎉 no goals
                                   -/
    else (le_of_dvd b0 h₁).antisymm (le_of_dvd a0 h₂)


instance isAntisymm : IsAntisymm Ordinal (· ∣ ·) :=
  ⟨@dvd_antisymm⟩


/-- `a % b` is the unique ordinal `o'` satisfying
  `a = b * o + o'` with `o' < b`. -/
instance mod : Mod Ordinal :=
  ⟨fun a b => a - b * (a / b)⟩


theorem mod_def (a b : Ordinal) : a % b = a - b * (a / b) :=
  rfl


theorem mod_le (a b : Ordinal) : a % b ≤ a :=
  sub_le_self a _


@[simp]
                                                 /-
                                                   a : Ordinal.{u_4}
                                                   ⊢ Eq (HMod.hMod a 0) a
                                                 -/
theorem mod_zero (a : Ordinal) : a % 0 = a := by simp only [mod_def, div_zero, zero_mul, sub_zero]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem mod_eq_of_lt {a b : Ordinal} (h : a < b) : a % b = a := by
  /-
    a b : Ordinal.{u_4}
    h : LT.lt a b
    ⊢ Eq (HMod.hMod a b) a
  -/
  simp only [mod_def, div_eq_zero_of_lt h, mul_zero, sub_zero]
  /-
    🎉 no goals
  -/


@[simp]
                                                 /-
                                                   b : Ordinal.{u_4}
                                                   ⊢ Eq (HMod.hMod 0 b) 0
                                                 -/
theorem zero_mod (b : Ordinal) : 0 % b = 0 := by simp only [mod_def, zero_div, mul_zero, sub_self]
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem div_add_mod (a b : Ordinal) : b * (a / b) + a % b = a :=
  Ordinal.add_sub_cancel_of_le <| mul_div_le _ _


theorem mod_lt (a) {b : Ordinal} (h : b ≠ 0) : a % b < b :=
                                              /-
                                                a b : Ordinal.{u_4}
                                                h : Ne b 0
                                                ⊢ LT.lt (HAdd.hAdd (HMul.hMul b (HDiv.hDiv a b)) (HMod.hMod a b)) (HAdd.hAdd ( …
                                              -/
  (add_lt_add_iff_left (b * (a / b))).1 <| by rw [div_add_mod]; exact lt_mul_div_add a h
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
theorem mod_self (a : Ordinal) : a % a = 0 :=
                        /-
                          a : Ordinal.{u_4}
                          a0 : Eq a 0
                          ⊢ Eq (HMod.hMod a a) 0
                        -/
  if a0 : a = 0 then by simp only [a0, zero_mod]
                        /-
                          🎉 no goals
                        -/
          /-
            a : Ordinal.{u_4}
            a0 : Not (Eq a 0)
            ⊢ Eq (HMod.hMod a a) 0
          -/
  else by simp only [mod_def, div_self a0, mul_one, sub_self]
          /-
            🎉 no goals
          -/


@[simp]
                                                /-
                                                  a : Ordinal.{u_4}
                                                  ⊢ Eq (HMod.hMod a 1) 0
                                                -/
theorem mod_one (a : Ordinal) : a % 1 = 0 := by simp only [mod_def, div_one, one_mul, sub_self]
                                                /-
                                                  🎉 no goals
                                                -/


theorem dvd_of_mod_eq_zero {a b : Ordinal} (H : a % b = 0) : b ∣ a :=
             /-
               a b : Ordinal.{u_4}
               H : Eq (HMod.hMod a b) 0
               ⊢ Eq a (HMul.hMul b (HDiv.hDiv a b))
             -/
  ⟨a / b, by simpa [H] using (div_add_mod a b).symm⟩
             /-
               🎉 no goals
             -/


theorem mod_eq_zero_of_dvd {a b : Ordinal} (H : b ∣ a) : a % b = 0 := by
  /-
    a b : Ordinal.{u_4}
    H : Dvd.dvd b a
    ⊢ Eq (HMod.hMod a b) 0
  -/
  rcases H with ⟨c, rfl⟩
  /-
    case intro
    b c : Ordinal.{u_4}
    ⊢ Eq (HMod.hMod (HMul.hMul b c) b) 0
  -/
  rcases eq_or_ne b 0 with (rfl | hb)
    /-
      case intro.inl
      c : Ordinal.{u_4}
      ⊢ Eq (HMod.hMod (HMul.hMul 0 c) 0) 0
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      b c : Ordinal.{u_4}
      hb : Ne b 0
      ⊢ Eq (HMod.hMod (HMul.hMul b c) b) 0
    -/
  · simp [mod_def, hb]
    /-
      🎉 no goals
    -/


theorem dvd_iff_mod_eq_zero {a b : Ordinal} : b ∣ a ↔ a % b = 0 :=
  ⟨mod_eq_zero_of_dvd, dvd_of_mod_eq_zero⟩


@[simp]
theorem mul_add_mod_self (x y z : Ordinal) : (x * y + z) % x = z % x := by
  /-
    x y z : Ordinal.{u_4}
    ⊢ Eq (HMod.hMod (HAdd.hAdd (HMul.hMul x y) z) x) (HMod.hMod z x)
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      y z : Ordinal.{u_4}
      ⊢ Eq (HMod.hMod (HAdd.hAdd (HMul.hMul 0 y) z) 0) (HMod.hMod z 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Ordinal.{u_4}
      hx : Ne x 0
      ⊢ Eq (HMod.hMod (HAdd.hAdd (HMul.hMul x y) z) x) (HMod.hMod z x)
    -/
  · rwa [mod_def, mul_add_div, mul_add, ← sub_sub, add_sub_cancel, mod_def]
    /-
      🎉 no goals
    -/


@[simp]
theorem mul_mod (x y : Ordinal) : x * y % x = 0 := by
  /-
    x y : Ordinal.{u_4}
    ⊢ Eq (HMod.hMod (HMul.hMul x y) x) 0
  -/
  simpa using mul_add_mod_self x y 0
  /-
    🎉 no goals
  -/


theorem mul_add_mod_mul {w x : Ordinal} (hw : w < x) (y z : Ordinal) :
    (x * y + w) % (x * z) = x * (y % z) + w := by
  /-
    w x : Ordinal.{u_4}
    hw : LT.lt w x
    y z : Ordinal.{u_4}
    ⊢ Eq (HMod.hMod (HAdd.hAdd (HMul.hMul x y) w) (HMul.hMul x z)) (HAdd.hAdd (HMu …
  -/
  rw [mod_def, mul_add_div_mul hw]
  /-
    w x : Ordinal.{u_4}
    hw : LT.lt w x
    y z : Ordinal.{u_4}
    ⊢ Eq (HSub.hSub (HAdd.hAdd (HMul.hMul x y) w) (HMul.hMul (HMul.hMul x z) (HDiv …
  -/
  apply sub_eq_of_add_eq
  /-
    case h
    w x : Ordinal.{u_4}
    hw : LT.lt w x
    y z : Ordinal.{u_4}
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HMul.hMul x z) (HDiv.hDiv y z)) (HAdd.hAdd (HMul.h …
  -/
  rw [← add_assoc, mul_assoc, ← mul_add, div_add_mod]
  /-
    🎉 no goals
  -/


theorem mul_mod_mul (x y z : Ordinal) : (x * y) % (x * z) = x * (y % z) := by
  /-
    x y z : Ordinal.{u_4}
    ⊢ Eq (HMod.hMod (HMul.hMul x y) (HMul.hMul x z)) (HMul.hMul x (HMod.hMod y z))
  -/
  obtain rfl | hx := Ordinal.eq_zero_or_pos x
    /-
      case inl
      y z : Ordinal.{u_4}
      ⊢ Eq (HMod.hMod (HMul.hMul 0 y) (HMul.hMul 0 z)) (HMul.hMul 0 (HMod.hMod y z))
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y z : Ordinal.{u_4}
      hx : LT.lt 0 x
      ⊢ Eq (HMod.hMod (HMul.hMul x y) (HMul.hMul x z)) (HMul.hMul x (HMod.hMod y z))
    -/
  · convert mul_add_mod_mul hx y z using 1 <;>
    /-
      case h.e'_2
      x y z : Ordinal.{u_4}
      hx : LT.lt 0 x
      ⊢ Eq (HMod.hMod (HMul.hMul x y) (HMul.hMul x z)) (HMod.hMod (HAdd.hAdd (HMul.h …
    -/
    /-
      🎉 no goals
    -/
    rw [add_zero]
    /-
      🎉 no goals
    -/


theorem mod_mod_of_dvd (a : Ordinal) {b c : Ordinal} (h : c ∣ b) : a % b % c = a % c := by
  /-
    a b c : Ordinal.{u_4}
    h : Dvd.dvd c b
    ⊢ Eq (HMod.hMod (HMod.hMod a b) c) (HMod.hMod a c)
  -/
  nth_rw 2 [← div_add_mod a b]
  /-
    a b c : Ordinal.{u_4}
    h : Dvd.dvd c b
    ⊢ Eq (HMod.hMod (HMod.hMod a b) c) (HMod.hMod (HAdd.hAdd (HMul.hMul b (HDiv.hD …
  -/
  rcases h with ⟨d, rfl⟩
  /-
    case intro
    a c d : Ordinal.{u_4}
    ⊢ Eq (HMod.hMod (HMod.hMod a (HMul.hMul c d)) c) (HMod.hMod (HAdd.hAdd (HMul.h …
  -/
  rw [mul_assoc, mul_add_mod_self]
  /-
    🎉 no goals
  -/


@[simp]
theorem mod_mod (a b : Ordinal) : a % b % b = a % b :=
  mod_mod_of_dvd a dvd_rfl


/-- Converts a family indexed by a `Type u` to one indexed by an `Ordinal.{u}` using a specified
well-ordering. -/
def bfamilyOfFamily' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] (f : ι → α) :
    ∀ a < type r, α := fun a ha => f (enum r ⟨a, ha⟩)


/-- Converts a family indexed by a `Type u` to one indexed by an `Ordinal.{u}` using a well-ordering
given by the axiom of choice. -/
def bfamilyOfFamily {ι : Type u} : (ι → α) → ∀ a < type (@WellOrderingRel ι), α :=
  bfamilyOfFamily' WellOrderingRel


/-- Converts a family indexed by an `Ordinal.{u}` to one indexed by a `Type u` using a specified
well-ordering. -/
def familyOfBFamily' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] {o} (ho : type r = o)
    (f : ∀ a < o, α) : ι → α := fun i =>
  f (typein r i)
    (by
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r✝ : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        ι : Type u
        r : ι → ι → Prop
        inst✝ : IsWellOrder ι r
        o : Ordinal.{u}
        ho : Eq (Ordinal.type r) o
        f : (a : Ordinal.{u}) → LT.lt a o → α
        i : ι
        ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding i) o
      -/
      rw [← ho]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        r✝ : α → α → Prop
        s : β → β → Prop
        t : γ → γ → Prop
        ι : Type u
        r : ι → ι → Prop
        inst✝ : IsWellOrder ι r
        o : Ordinal.{u}
        ho : Eq (Ordinal.type r) o
        f : (a : Ordinal.{u}) → LT.lt a o → α
        i : ι
        ⊢ LT.lt ((Ordinal.typein r).toRelEmbedding i) (Ordinal.type r)
      -/
      exact typein_lt_type r i)
      /-
        🎉 no goals
      -/


/-- Converts a family indexed by an `Ordinal.{u}` to one indexed by a `Type u` using a well-ordering
given by the axiom of choice. -/
def familyOfBFamily (o : Ordinal) (f : ∀ a < o, α) : o.toType → α :=
  familyOfBFamily' (· < ·) (type_toType o) f


@[simp]
theorem bfamilyOfFamily'_typein {ι} (r : ι → ι → Prop) [IsWellOrder ι r] (f : ι → α) (i) :
    bfamilyOfFamily' r f (typein r i) (typein_lt_type r i) = f i := by
  /-
    α : Type u_1
    ι : Type u_4
    r : ι → ι → Prop
    inst✝ : IsWellOrder ι r
    f : ι → α
    i : ι
    ⊢ Eq (Ordinal.bfamilyOfFamily' r f ((Ordinal.typein r).toRelEmbedding i) ⋯) (f …
  -/
  simp only [bfamilyOfFamily', enum_typein]
  /-
    🎉 no goals
  -/


@[simp]
theorem bfamilyOfFamily_typein {ι} (f : ι → α) (i) :
    bfamilyOfFamily f (typein _ i) (typein_lt_type _ i) = f i :=
  bfamilyOfFamily'_typein _ f i


theorem familyOfBFamily'_enum {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] {o}
    (ho : type r = o) (f : ∀ a < o, α) (i hi) :
                                           /-
                                             α : Type u_1
                                             β : Type u_2
                                             γ : Type u_3
                                             r✝ : α → α → Prop
                                             s : β → β → Prop
                                             t : γ → γ → Prop
                                             ι : Type u
                                             r : ι → ι → Prop
                                             inst✝ : IsWellOrder ι r
                                             o : Ordinal.{u}
                                             ho : Eq (Ordinal.type r) o
                                             f : (a : Ordinal.{u}) → LT.lt a o → α
                                             i : Ordinal.{u}
                                             hi : LT.lt i o
                                             ⊢ LT.lt i (Ordinal.type r)
                                           -/
    familyOfBFamily' r ho f (enum r ⟨i, by rwa [ho]⟩) = f i hi := by
                                           /-
                                             🎉 no goals
                                           -/
  /-
    α : Type u_1
    ι : Type u
    r : ι → ι → Prop
    inst✝ : IsWellOrder ι r
    o : Ordinal.{u}
    ho : Eq (Ordinal.type r) o
    f : (a : Ordinal.{u}) → LT.lt a o → α
    i : Ordinal.{u}
    hi : LT.lt i o
    ⊢ Eq (Ordinal.familyOfBFamily' r ho f ((Ordinal.enum r) ⟨i, ⋯⟩)) (f i hi)
  -/
  simp only [familyOfBFamily', typein_enum]
  /-
    🎉 no goals
  -/


theorem familyOfBFamily_enum (o : Ordinal) (f : ∀ a < o, α) (i hi) :
    familyOfBFamily o f (enum (α := o.toType) (· < ·) ⟨i, hi.trans_eq (type_toType _).symm⟩)
    = f i hi :=
  familyOfBFamily'_enum _ (type_toType o) f _ _


/-- The range of a family indexed by ordinals. -/
def brange (o : Ordinal) (f : ∀ a < o, α) : Set α :=
  { a | ∃ i hi, f i hi = a }


theorem mem_brange {o : Ordinal} {f : ∀ a < o, α} {a} : a ∈ brange o f ↔ ∃ i hi, f i hi = a :=
  Iff.rfl


theorem mem_brange_self {o} (f : ∀ a < o, α) (i hi) : f i hi ∈ brange o f :=
  ⟨i, hi, rfl⟩


@[simp]
theorem range_familyOfBFamily' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] {o}
    (ho : type r = o) (f : ∀ a < o, α) : range (familyOfBFamily' r ho f) = brange o f := by
  /-
    α : Type u_1
    ι : Type u
    r : ι → ι → Prop
    inst✝ : IsWellOrder ι r
    o : Ordinal.{u}
    ho : Eq (Ordinal.type r) o
    f : (a : Ordinal.{u}) → LT.lt a o → α
    ⊢ Eq (Set.range (Ordinal.familyOfBFamily' r ho f)) (o.brange f)
  -/
  refine Set.ext fun a => ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      o : Ordinal.{u}
      ho : Eq (Ordinal.type r) o
      f : (a : Ordinal.{u}) → LT.lt a o → α
      a : α
      ⊢ Membership.mem (Set.range (Ordinal.familyOfBFamily' r ho f)) a → Membership. …
    -/
  · rintro ⟨b, rfl⟩
    /-
      case refine_1.intro
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      o : Ordinal.{u}
      ho : Eq (Ordinal.type r) o
      f : (a : Ordinal.{u}) → LT.lt a o → α
      b : ι
      ⊢ Membership.mem (o.brange f) (Ordinal.familyOfBFamily' r ho f b)
    -/
    apply mem_brange_self
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      o : Ordinal.{u}
      ho : Eq (Ordinal.type r) o
      f : (a : Ordinal.{u}) → LT.lt a o → α
      a : α
      ⊢ Membership.mem (o.brange f) a → Membership.mem (Set.range (Ordinal.familyOfB …
    -/
  · rintro ⟨i, hi, rfl⟩
    /-
      case refine_2.intro.intro
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      o : Ordinal.{u}
      ho : Eq (Ordinal.type r) o
      f : (a : Ordinal.{u}) → LT.lt a o → α
      i : Ordinal.{u}
      hi : LT.lt i o
      ⊢ Membership.mem (Set.range (Ordinal.familyOfBFamily' r ho f)) (f i hi)
    -/
    exact ⟨_, familyOfBFamily'_enum _ _ _ _ _⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem range_familyOfBFamily {o} (f : ∀ a < o, α) : range (familyOfBFamily o f) = brange o f :=
  range_familyOfBFamily' _ _ f


@[simp]
theorem brange_bfamilyOfFamily' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] (f : ι → α) :
    brange _ (bfamilyOfFamily' r f) = range f := by
  /-
    α : Type u_1
    ι : Type u
    r : ι → ι → Prop
    inst✝ : IsWellOrder ι r
    f : ι → α
    ⊢ Eq ((Ordinal.type r).brange (Ordinal.bfamilyOfFamily' r f)) (Set.range f)
  -/
  refine Set.ext fun a => ⟨?_, ?_⟩
    /-
      case refine_1
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      f : ι → α
      a : α
      ⊢ Membership.mem ((Ordinal.type r).brange (Ordinal.bfamilyOfFamily' r f)) a →  …
    -/
  · rintro ⟨i, hi, rfl⟩
    /-
      case refine_1.intro.intro
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      f : ι → α
      i : Ordinal.{u}
      hi : LT.lt i (Ordinal.type r)
      ⊢ Membership.mem (Set.range f) (Ordinal.bfamilyOfFamily' r f i hi)
    -/
    apply mem_range_self
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      f : ι → α
      a : α
      ⊢ Membership.mem (Set.range f) a → Membership.mem ((Ordinal.type r).brange (Or …
    -/
  · rintro ⟨b, rfl⟩
    /-
      case refine_2.intro
      α : Type u_1
      ι : Type u
      r : ι → ι → Prop
      inst✝ : IsWellOrder ι r
      f : ι → α
      b : ι
      ⊢ Membership.mem ((Ordinal.type r).brange (Ordinal.bfamilyOfFamily' r f)) (f b)
    -/
    exact ⟨_, _, bfamilyOfFamily'_typein _ _ _⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem brange_bfamilyOfFamily {ι : Type u} (f : ι → α) : brange _ (bfamilyOfFamily f) = range f :=
  brange_bfamilyOfFamily' _ _


@[simp]
theorem brange_const {o : Ordinal} (ho : o ≠ 0) {c : α} : (brange o fun _ _ => c) = {c} := by
  /-
    α : Type u_1
    o : Ordinal.{u_4}
    ho : Ne o 0
    c : α
    ⊢ Eq (o.brange fun x x => c) (Singleton.singleton c)
  -/
  rw [← range_familyOfBFamily]
  /-
    α : Type u_1
    o : Ordinal.{u_4}
    ho : Ne o 0
    c : α
    ⊢ Eq (Set.range (o.familyOfBFamily fun x x => c)) (Singleton.singleton c)
  -/
  exact @Set.range_const _ o.toType (toType_nonempty_iff_ne_zero.2 ho) c
  /-
    🎉 no goals
  -/


theorem comp_bfamilyOfFamily' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] (f : ι → α)
    (g : α → β) : (fun i hi => g (bfamilyOfFamily' r f i hi)) = bfamilyOfFamily' r (g ∘ f) :=
  rfl


theorem comp_bfamilyOfFamily {ι : Type u} (f : ι → α) (g : α → β) :
    (fun i hi => g (bfamilyOfFamily f i hi)) = bfamilyOfFamily (g ∘ f) :=
  rfl


theorem comp_familyOfBFamily' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] {o}
    (ho : type r = o) (f : ∀ a < o, α) (g : α → β) :
    g ∘ familyOfBFamily' r ho f = familyOfBFamily' r ho fun i hi => g (f i hi) :=
  rfl


theorem comp_familyOfBFamily {o} (f : ∀ a < o, α) (g : α → β) :
    g ∘ familyOfBFamily o f = familyOfBFamily o fun i hi => g (f i hi) :=
  rfl


/-- The supremum of a family of ordinals -/

@[deprecated iSup (since := "2024-08-27")]
def sup {ι : Type u} (f : ι → Ordinal.{max u v}) : Ordinal.{max u v} :=
  iSup f


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-27")]
theorem sSup_eq_sup {ι : Type u} (f : ι → Ordinal.{max u v}) : sSup (Set.range f) = sup.{_, v} f :=
  rfl


/-- The range of an indexed ordinal function, whose outputs live in a higher universe than the
    inputs, is always bounded above. See `Ordinal.lsub` for an explicit bound. -/
theorem bddAbove_range {ι : Type u} (f : ι → Ordinal.{max u v}) : BddAbove (Set.range f) :=
  ⟨(iSup (succ ∘ card ∘ f)).ord, by
    /-
      ι : Type u
      f : ι → Ordinal.{max u v}
      ⊢ Membership.mem (upperBounds (Set.range f)) (iSup (Function.comp Order.succ ( …
    -/
    rintro a ⟨i, rfl⟩
    exact le_of_lt (Cardinal.lt_ord.2 ((lt_succ _).trans_le
      (le_ciSup (Cardinal.bddAbove_range _) _)))⟩


theorem bddAbove_of_small (s : Set Ordinal.{u}) [h : Small.{u} s] : BddAbove s := by
  /-
    s : Set Ordinal.{u}
    h : Small.{u, u + 1} ↑s
    ⊢ BddAbove s
  -/
  obtain ⟨a, ha⟩ := bddAbove_range (fun x => ((@equivShrink s h).symm x).val)
  /-
    case intro
    s : Set Ordinal.{u}
    h : Small.{u, u + 1} ↑s
    a : Ordinal.{u}
    ha : Membership.mem (upperBounds (Set.range fun x => ↑((equivShrink ↑s).symm x …
    ⊢ BddAbove s
  -/
  use a
  /-
    case h
    s : Set Ordinal.{u}
    h : Small.{u, u + 1} ↑s
    a : Ordinal.{u}
    ha : Membership.mem (upperBounds (Set.range fun x => ↑((equivShrink ↑s).symm x …
    ⊢ Membership.mem (upperBounds s) a
  -/
  intro b hb
  /-
    case h
    s : Set Ordinal.{u}
    h : Small.{u, u + 1} ↑s
    a : Ordinal.{u}
    ha : Membership.mem (upperBounds (Set.range fun x => ↑((equivShrink ↑s).symm x …
    b : Ordinal.{u}
    hb : Membership.mem s b
    ⊢ LE.le b a
  -/
  simpa using ha (mem_range_self (equivShrink s ⟨b, hb⟩))
  /-
    🎉 no goals
  -/


theorem bddAbove_iff_small {s : Set Ordinal.{u}} : BddAbove s ↔ Small.{u} s :=
  ⟨fun ⟨a, h⟩ => small_subset <| show s ⊆ Iic a from fun _ hx => h hx, fun _ =>
    bddAbove_of_small _⟩


theorem bddAbove_image {s : Set Ordinal.{u}} (hf : BddAbove s)
    (f : Ordinal.{u} → Ordinal.{max u v}) : BddAbove (f '' s) := by
  /-
    s : Set Ordinal.{u}
    hf : BddAbove s
    f : Ordinal.{u} → Ordinal.{max u v}
    ⊢ BddAbove (Set.image f s)
  -/
  rw [bddAbove_iff_small] at hf ⊢
  /-
    s : Set Ordinal.{u}
    hf : Small.{u, u + 1} ↑s
    f : Ordinal.{u} → Ordinal.{max u v}
    ⊢ Small.{max u v, (max u v) + 1} ↑(Set.image f s)
  -/
  exact small_lift _
  /-
    🎉 no goals
  -/


theorem bddAbove_range_comp {ι : Type u} {f : ι → Ordinal.{v}} (hf : BddAbove (range f))
    (g : Ordinal.{v} → Ordinal.{max v w}) : BddAbove (range (g ∘ f)) := by
  /-
    ι : Type u
    f : ι → Ordinal.{v}
    hf : BddAbove (Set.range f)
    g : Ordinal.{v} → Ordinal.{max v w}
    ⊢ BddAbove (Set.range (Function.comp g f))
  -/
  rw [range_comp]
  /-
    ι : Type u
    f : ι → Ordinal.{v}
    hf : BddAbove (Set.range f)
    g : Ordinal.{v} → Ordinal.{max v w}
    ⊢ BddAbove (Set.image g (Set.range f))
  -/
  exact bddAbove_image hf g
  /-
    🎉 no goals
  -/


/-- `le_ciSup` whenever the input type is small in the output universe. This lemma sometimes
fails to infer `f` in simple cases and needs it to be given explicitly. -/
protected theorem le_iSup {ι} (f : ι → Ordinal.{u}) [Small.{u} ι] : ∀ i, f i ≤ iSup f :=
  le_ciSup (bddAbove_of_small _)


set_option linter.deprecated false in
@[deprecated Ordinal.le_iSup (since := "2024-08-27")]
theorem le_sup {ι : Type u} (f : ι → Ordinal.{max u v}) : ∀ i, f i ≤ sup.{_, v} f := fun i =>
  Ordinal.le_iSup f i


/-- `ciSup_le_iff'` whenever the input type is small in the output universe. -/
protected theorem iSup_le_iff {ι} {f : ι → Ordinal.{u}} {a : Ordinal.{u}} [Small.{u} ι] :
    iSup f ≤ a ↔ ∀ i, f i ≤ a :=
  ciSup_le_iff' (bddAbove_of_small _)


set_option linter.deprecated false in
@[deprecated Ordinal.iSup_le_iff (since := "2024-08-27")]
theorem sup_le_iff {ι : Type u} {f : ι → Ordinal.{max u v}} {a} : sup.{_, v} f ≤ a ↔ ∀ i, f i ≤ a :=
  Ordinal.iSup_le_iff


/-- An alias of `ciSup_le'` for discoverability. -/
protected theorem iSup_le {ι} {f : ι → Ordinal} {a} :
    (∀ i, f i ≤ a) → iSup f ≤ a :=
  ciSup_le'


set_option linter.deprecated false in
@[deprecated Ordinal.iSup_le (since := "2024-08-27")]
theorem sup_le {ι : Type u} {f : ι → Ordinal.{max u v}} {a} : (∀ i, f i ≤ a) → sup.{_, v} f ≤ a :=
  Ordinal.iSup_le


/-- `lt_ciSup_iff'` whenever the input type is small in the output universe. -/
protected theorem lt_iSup_iff {ι} {f : ι → Ordinal.{u}} {a : Ordinal.{u}} [Small.{u} ι] :
    a < iSup f ↔ ∃ i, a < f i :=
  lt_ciSup_iff' (bddAbove_of_small _)


@[deprecated "No deprecation message was provided." (since := "2024-11-12")]
alias lt_iSup := lt_iSup_iff


set_option linter.deprecated false in
@[deprecated Ordinal.lt_iSup (since := "2024-08-27")]
theorem lt_sup {ι : Type u} {f : ι → Ordinal.{max u v}} {a} : a < sup.{_, v} f ↔ ∃ i, a < f i := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    a : Ordinal.{max u v}
    ⊢ Iff (LT.lt a (Ordinal.sup f)) (Exists fun i => LT.lt a (f i))
  -/
  simpa only [not_forall, not_le] using not_congr (@sup_le_iff.{_, v} _ f a)
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-08-27")]
theorem ne_iSup_iff_lt_iSup {ι : Type u} {f : ι → Ordinal.{max u v}} :
    (∀ i, f i ≠ iSup f) ↔ ∀ i, f i < iSup f :=
  forall_congr' fun i => (Ordinal.le_iSup f i).lt_iff_ne.symm


set_option linter.deprecated false in
@[deprecated ne_iSup_iff_lt_iSup (since := "2024-08-27")]
theorem ne_sup_iff_lt_sup {ι : Type u} {f : ι → Ordinal.{max u v}} :
    (∀ i, f i ≠ sup.{_, v} f) ↔ ∀ i, f i < sup.{_, v} f :=
  ne_iSup_iff_lt_iSup

-- TODO: state in terms of `IsSuccLimit`.

theorem succ_lt_iSup_of_ne_iSup {ι} {f : ι → Ordinal.{u}} [Small.{u} ι]
    (hf : ∀ i, f i ≠ iSup f) {a} (hao : a < iSup f) : succ a < iSup f := by
  /-
    ι : Type u_4
    f : ι → Ordinal.{u}
    inst✝ : Small.{u, u_4} ι
    hf : ∀ (i : ι), Ne (f i) (iSup f)
    a : Ordinal.{u}
    hao : LT.lt a (iSup f)
    ⊢ LT.lt (Order.succ a) (iSup f)
  -/
  by_contra! hoa
  exact hao.not_le (Ordinal.iSup_le fun i => le_of_lt_succ <|
    (lt_of_le_of_ne (Ordinal.le_iSup _ _) (hf i)).trans_le hoa)


set_option linter.deprecated false in
@[deprecated succ_lt_iSup_of_ne_iSup (since := "2024-08-27")]
theorem sup_not_succ_of_ne_sup {ι : Type u} {f : ι → Ordinal.{max u v}}
    (hf : ∀ i, f i ≠ sup.{_, v} f) {a} (hao : a < sup.{_, v} f) : succ a < sup.{_, v} f := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    hf : ∀ (i : ι), Ne (f i) (Ordinal.sup f)
    a : Ordinal.{max u v}
    hao : LT.lt a (Ordinal.sup f)
    ⊢ LT.lt (Order.succ a) (Ordinal.sup f)
  -/
  by_contra! hoa
  exact
    hao.not_le (sup_le fun i => le_of_lt_succ <| (lt_of_le_of_ne (le_sup _ _) (hf i)).trans_le hoa)

-- TODO: generalize to conditionally complete lattices.

theorem iSup_eq_zero_iff {ι} {f : ι → Ordinal.{u}} [Small.{u} ι] :
    iSup f = 0 ↔ ∀ i, f i = 0 := by
  refine
    ⟨fun h i => ?_, fun h =>
      le_antisymm (Ordinal.iSup_le fun i => Ordinal.le_zero.2 (h i)) (Ordinal.zero_le _)⟩
  /-
    ι : Type u_4
    f : ι → Ordinal.{u}
    inst✝ : Small.{u, u_4} ι
    h : Eq (iSup f) 0
    i : ι
    ⊢ Eq (f i) 0
  -/
  rw [← Ordinal.le_zero, ← h]
  /-
    ι : Type u_4
    f : ι → Ordinal.{u}
    inst✝ : Small.{u, u_4} ι
    h : Eq (iSup f) 0
    i : ι
    ⊢ LE.le (f i) (iSup f)
  -/
  exact Ordinal.le_iSup f i
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated iSup_eq_zero_iff (since := "2024-08-27")]
theorem sup_eq_zero_iff {ι : Type u} {f : ι → Ordinal.{max u v}} :
    sup.{_, v} f = 0 ↔ ∀ i, f i = 0 := by
  refine
    ⟨fun h i => ?_, fun h =>
      le_antisymm (sup_le fun i => Ordinal.le_zero.2 (h i)) (Ordinal.zero_le _)⟩
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    h : Eq (Ordinal.sup f) 0
    i : ι
    ⊢ Eq (f i) 0
  -/
  rw [← Ordinal.le_zero, ← h]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    h : Eq (Ordinal.sup f) 0
    i : ι
    ⊢ LE.le (f i) (Ordinal.sup f)
  -/
  exact le_sup f i
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated ciSup_of_empty (since := "2024-08-27")]
theorem sup_empty {ι} [IsEmpty ι] (f : ι → Ordinal) : sup f = 0 :=
  ciSup_of_empty f


set_option linter.deprecated false in
@[deprecated ciSup_const (since := "2024-08-27")]
theorem sup_const {ι} [_hι : Nonempty ι] (o : Ordinal) : (sup fun _ : ι => o) = o :=
  ciSup_const


set_option linter.deprecated false in
@[deprecated ciSup_unique (since := "2024-08-27")]
theorem sup_unique {ι} [Unique ι] (f : ι → Ordinal) : sup f = f default :=
  ciSup_unique


set_option linter.deprecated false in
@[deprecated csSup_le_csSup' (since := "2024-08-27")]
theorem sup_le_of_range_subset {ι ι'} {f : ι → Ordinal} {g : ι' → Ordinal}
    (h : Set.range f ⊆ Set.range g) : sup.{u, max v w} f ≤ sup.{v, max u w} g :=
  csSup_le_csSup' (bddAbove_range.{v, max u w} _) h

-- TODO: generalize or remove

theorem iSup_eq_of_range_eq {ι ι'} {f : ι → Ordinal} {g : ι' → Ordinal}
    (h : Set.range f = Set.range g) : iSup f = iSup g :=
  congr_arg _ h


set_option linter.deprecated false in
@[deprecated iSup_eq_of_range_eq (since := "2024-08-27")]
theorem sup_eq_of_range_eq {ι : Type u} {ι' : Type v}
    {f : ι → Ordinal.{max u v w}} {g : ι' → Ordinal.{max u v w}}
    (h : Set.range f = Set.range g) : sup.{u, max v w} f = sup.{v, max u w} g :=
  Ordinal.iSup_eq_of_range_eq h

-- TODO: generalize to conditionally complete lattices

theorem iSup_sum {α β} (f : α ⊕ β → Ordinal.{u}) [Small.{u} α] [Small.{u} β]:
    iSup f = max (⨆ a, f (Sum.inl a)) (⨆ b, f (Sum.inr b)) := by
  /-
    α : Type u_4
    β : Type u_5
    f : Sum α β → Ordinal.{u}
    inst✝¹ : Small.{u, u_4} α
    inst✝ : Small.{u, u_5} β
    ⊢ Eq (iSup f) (Max.max (iSup fun a => f (Sum.inl a)) (iSup fun b => f (Sum.inr …
  -/
  apply (Ordinal.iSup_le _).antisymm (max_le _ _)
    /-
      α : Type u_4
      β : Type u_5
      f : Sum α β → Ordinal.{u}
      inst✝¹ : Small.{u, u_4} α
      inst✝ : Small.{u, u_5} β
      ⊢ ∀ (i : Sum α β), LE.le (f i) (Max.max (iSup fun a => f (Sum.inl a)) (iSup fu …
    -/
  · rintro (i | i)
      /-
        case inl
        α : Type u_4
        β : Type u_5
        f : Sum α β → Ordinal.{u}
        inst✝¹ : Small.{u, u_4} α
        inst✝ : Small.{u, u_5} β
        i : α
        ⊢ LE.le (f (Sum.inl i)) (Max.max (iSup fun a => f (Sum.inl a)) (iSup fun b =>  …
      -/
    · exact le_max_of_le_left (Ordinal.le_iSup (fun x ↦ f (Sum.inl x)) i)
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u_4
        β : Type u_5
        f : Sum α β → Ordinal.{u}
        inst✝¹ : Small.{u, u_4} α
        inst✝ : Small.{u, u_5} β
        i : β
        ⊢ LE.le (f (Sum.inr i)) (Max.max (iSup fun a => f (Sum.inl a)) (iSup fun b =>  …
      -/
    · exact le_max_of_le_right (Ordinal.le_iSup (fun x ↦ f (Sum.inr x)) i)
      /-
        🎉 no goals
      -/
  all_goals
    apply csSup_le_csSup' (bddAbove_of_small _)
    rintro i ⟨a, rfl⟩
    apply mem_range_self


set_option linter.deprecated false in
@[deprecated iSup_sum (since := "2024-08-27")]
theorem sup_sum {α : Type u} {β : Type v} (f : α ⊕ β → Ordinal) :
    sup.{max u v, w} f =
      max (sup.{u, max v w} fun a => f (Sum.inl a)) (sup.{v, max u w} fun b => f (Sum.inr b)) := by
  /-
    α : Type u
    β : Type v
    f : Sum α β → Ordinal.{max (max u v) w}
    ⊢ Eq (Ordinal.sup f) (Max.max (Ordinal.sup fun a => f (Sum.inl a)) (Ordinal.su …
  -/
  apply (sup_le_iff.2 _).antisymm (max_le_iff.2 ⟨_, _⟩)
    /-
      α : Type u
      β : Type v
      f : Sum α β → Ordinal.{max (max u v) w}
      ⊢ ∀ (i : Sum α β), LE.le (f i) (Max.max (Ordinal.sup fun a => f (Sum.inl a)) ( …
    -/
  · rintro (i | i)
      /-
        case inl
        α : Type u
        β : Type v
        f : Sum α β → Ordinal.{max (max u v) w}
        i : α
        ⊢ LE.le (f (Sum.inl i)) (Max.max (Ordinal.sup fun a => f (Sum.inl a)) (Ordinal …
      -/
    · exact le_max_of_le_left (le_sup _ i)
      /-
        🎉 no goals
      -/
      /-
        case inr
        α : Type u
        β : Type v
        f : Sum α β → Ordinal.{max (max u v) w}
        i : β
        ⊢ LE.le (f (Sum.inr i)) (Max.max (Ordinal.sup fun a => f (Sum.inl a)) (Ordinal …
      -/
    · exact le_max_of_le_right (le_sup _ i)
      /-
        🎉 no goals
      -/
  all_goals
    apply sup_le_of_range_subset.{_, max u v, w}
    rintro i ⟨a, rfl⟩
    apply mem_range_self


theorem unbounded_range_of_le_iSup {α β : Type u} (r : α → α → Prop) [IsWellOrder α r] (f : β → α)
    (h : type r ≤ ⨆ i, typein r (f i)) : Unbounded r (range f) :=
  (not_bounded_iff _).1 fun ⟨x, hx⟩ =>
    h.not_lt <| lt_of_le_of_lt
      (Ordinal.iSup_le fun y => ((typein_lt_typein r).2 <| hx _ <| mem_range_self y).le)
      (typein_lt_type r x)


set_option linter.deprecated false in
@[deprecated unbounded_range_of_le_iSup (since := "2024-08-27")]
theorem unbounded_range_of_sup_ge {α β : Type u} (r : α → α → Prop) [IsWellOrder α r] (f : β → α)
    (h : type r ≤ sup.{u, u} (typein r ∘ f)) : Unbounded r (range f) :=
  unbounded_range_of_le_iSup r f h


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-27")]
theorem le_sup_shrink_equiv {s : Set Ordinal.{u}} (hs : Small.{u} s) (a) (ha : a ∈ s) :
    a ≤ sup.{u, u} fun x => ((@equivShrink s hs).symm x).val := by
  /-
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    a : Ordinal.{u}
    ha : Membership.mem s a
    ⊢ LE.le a (Ordinal.sup fun x => ↑((equivShrink ↑s).symm x))
  -/
  convert le_sup.{u, u} (fun x => ((@equivShrink s hs).symm x).val) ((@equivShrink s hs) ⟨a, ha⟩)
  /-
    case h.e'_3
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    a : Ordinal.{u}
    ha : Membership.mem s a
    ⊢ Eq a ↑((equivShrink ↑s).symm ((equivShrink ↑s) ⟨a, ha⟩))
  -/
  rw [symm_apply_apply]
  /-
    🎉 no goals
  -/


theorem IsNormal.map_iSup_of_bddAbove {f : Ordinal.{u} → Ordinal.{v}} (H : IsNormal f)
    {ι : Type*} (g : ι → Ordinal.{u}) (hg : BddAbove (range g))
    [Nonempty ι] : f (⨆ i, g i) = ⨆ i, f (g i) := eq_of_forall_ge_iff fun a ↦ by
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    ι : Type u_4
    g : ι → Ordinal.{u}
    hg : BddAbove (Set.range g)
    inst✝ : Nonempty ι
    a : Ordinal.{v}
    ⊢ Iff (LE.le (f (iSup fun i => g i)) a) (LE.le (iSup fun i => f (g i)) a)
  -/
  have := bddAbove_iff_small.mp hg
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    ι : Type u_4
    g : ι → Ordinal.{u}
    hg : BddAbove (Set.range g)
    inst✝ : Nonempty ι
    a : Ordinal.{v}
    this : Small.{u, u + 1} ↑(Set.range g)
    ⊢ Iff (LE.le (f (iSup fun i => g i)) a) (LE.le (iSup fun i => f (g i)) a)
  -/
  have := univLE_of_injective H.strictMono.injective
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    ι : Type u_4
    g : ι → Ordinal.{u}
    hg : BddAbove (Set.range g)
    inst✝ : Nonempty ι
    a : Ordinal.{v}
    this✝ : Small.{u, u + 1} ↑(Set.range g)
    this : UnivLE.{u, v}
    ⊢ Iff (LE.le (f (iSup fun i => g i)) a) (LE.le (iSup fun i => f (g i)) a)
  -/
  have := Small.trans_univLE.{u, v} (range g)
  have hfg : BddAbove (range (f ∘ g)) := bddAbove_iff_small.mpr <| by
    rw [range_comp]
    exact small_image f (range g)
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    ι : Type u_4
    g : ι → Ordinal.{u}
    hg : BddAbove (Set.range g)
    inst✝ : Nonempty ι
    a : Ordinal.{v}
    this✝¹ : Small.{u, u + 1} ↑(Set.range g)
    this✝ : UnivLE.{u, v}
    this : Small.{v, u + 1} ↑(Set.range g)
    hfg : BddAbove (Set.range (Function.comp f g))
    ⊢ Iff (LE.le (f (iSup fun i => g i)) a) (LE.le (iSup fun i => f (g i)) a)
  -/
  change _ ↔ ⨆ i, (f ∘ g) i ≤ a
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    ι : Type u_4
    g : ι → Ordinal.{u}
    hg : BddAbove (Set.range g)
    inst✝ : Nonempty ι
    a : Ordinal.{v}
    this✝¹ : Small.{u, u + 1} ↑(Set.range g)
    this✝ : UnivLE.{u, v}
    this : Small.{v, u + 1} ↑(Set.range g)
    hfg : BddAbove (Set.range (Function.comp f g))
    ⊢ Iff (LE.le (f (iSup fun i => g i)) a) (LE.le (iSup fun i => Function.comp f  …
  -/
                                                             /-
                                                               🎉 no goals
                                                             -/
  rw [ciSup_le_iff hfg, H.le_set' _ Set.univ_nonempty g] <;> simp [ciSup_le_iff hg]
                                                             /-
                                                               🎉 no goals
                                                             -/


theorem IsNormal.map_iSup {f : Ordinal.{u} → Ordinal.{v}} (H : IsNormal f)
    {ι : Type w} (g : ι → Ordinal.{u}) [Small.{u} ι] [Nonempty ι] :
    f (⨆ i, g i) = ⨆ i, f (g i) :=
  H.map_iSup_of_bddAbove g (bddAbove_of_small _)


theorem IsNormal.map_sSup_of_bddAbove {f : Ordinal.{u} → Ordinal.{v}} (H : IsNormal f)
    {s : Set Ordinal.{u}} (hs : BddAbove s) (hn : s.Nonempty) : f (sSup s) = sSup (f '' s) := by
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    s : Set Ordinal.{u}
    hs : BddAbove s
    hn : s.Nonempty
    ⊢ Eq (f (SupSet.sSup s)) (SupSet.sSup (Set.image f s))
  -/
  have := hn.to_subtype
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    s : Set Ordinal.{u}
    hs : BddAbove s
    hn : s.Nonempty
    this : Nonempty ↑s
    ⊢ Eq (f (SupSet.sSup s)) (SupSet.sSup (Set.image f s))
  -/
  rw [sSup_eq_iSup', sSup_image', H.map_iSup_of_bddAbove]
  /-
    case hg
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    s : Set Ordinal.{u}
    hs : BddAbove s
    hn : s.Nonempty
    this : Nonempty ↑s
    ⊢ BddAbove (Set.range Subtype.val)
  -/
  rwa [Subtype.range_coe_subtype, setOf_mem_eq]
  /-
    🎉 no goals
  -/


theorem IsNormal.map_sSup {f : Ordinal.{u} → Ordinal.{v}} (H : IsNormal f)
    {s : Set Ordinal.{u}} (hn : s.Nonempty) [Small.{u} s] : f (sSup s) = sSup (f '' s) :=
  H.map_sSup_of_bddAbove (bddAbove_of_small s) hn


set_option linter.deprecated false in
@[deprecated IsNormal.map_iSup (since := "2024-08-27")]
theorem IsNormal.sup {f : Ordinal.{max u v} → Ordinal.{max u w}} (H : IsNormal f) {ι : Type u}
    (g : ι → Ordinal.{max u v}) [Nonempty ι] : f (sup.{_, v} g) = sup.{_, w} (f ∘ g) :=
  H.map_iSup g


theorem IsNormal.apply_of_isLimit {f : Ordinal.{u} → Ordinal.{v}} (H : IsNormal f) {o : Ordinal}
    (ho : IsLimit o) : f o = ⨆ a : Iio o, f a := by
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    o : Ordinal.{u}
    ho : o.IsLimit
    ⊢ Eq (f o) (iSup fun a => f ↑a)
  -/
  have : Nonempty (Iio o) := ⟨0, ho.pos⟩
  /-
    f : Ordinal.{u} → Ordinal.{v}
    H : Ordinal.IsNormal f
    o : Ordinal.{u}
    ho : o.IsLimit
    this : Nonempty ↑(Set.Iio o)
    ⊢ Eq (f o) (iSup fun a => f ↑a)
  -/
  rw [← H.map_iSup, ho.iSup_Iio]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-08-27")]
theorem sup_eq_sSup {s : Set Ordinal.{u}} (hs : Small.{u} s) :
    (sup.{u, u} fun x => (@equivShrink s hs).symm x) = sSup s :=
  let hs' := bddAbove_iff_small.2 hs
  ((csSup_le_iff' hs').2 (le_sup_shrink_equiv hs)).antisymm'
    (sup_le fun _x => le_csSup hs' (Subtype.mem _))


theorem sSup_ord {s : Set Cardinal.{u}} (hs : BddAbove s) : (sSup s).ord = sSup (ord '' s) :=
  eq_of_forall_ge_iff fun a => by
    rw [csSup_le_iff'
        (bddAbove_iff_small.2 (@small_image _ _ _ s (Cardinal.bddAbove_iff_small.1 hs))),
      ord_le, csSup_le_iff' hs]
    /-
      s : Set Cardinal.{u}
      hs : BddAbove s
      a : Ordinal.{u}
      ⊢ Iff (∀ (x : Cardinal.{u}), Membership.mem s x → LE.le x a.card) (∀ (x : Ordi …
    -/
    simp [ord_le]
    /-
      🎉 no goals
    -/


theorem iSup_ord {ι} {f : ι → Cardinal} (hf : BddAbove (range f)) :
    (iSup f).ord = ⨆ i, (f i).ord := by
  /-
    ι : Sort u_4
    f : ι → Cardinal.{u_5}
    hf : BddAbove (Set.range f)
    ⊢ Eq (iSup f).ord (iSup fun i => (f i).ord)
  -/
  unfold iSup
  /-
    ι : Sort u_4
    f : ι → Cardinal.{u_5}
    hf : BddAbove (Set.range f)
    ⊢ Eq (SupSet.sSup (Set.range f)).ord (SupSet.sSup (Set.range fun i => (f i).or …
  -/
  convert sSup_ord hf
  -- Porting note: `change` is required.
  /-
    case h.e'_3.h.e'_3
    ι : Sort u_4
    f : ι → Cardinal.{u_5}
    hf : BddAbove (Set.range f)
    ⊢ Eq (Set.range fun i => (f i).ord) (Set.image Cardinal.ord (Set.range f))
  -/
  conv_lhs => change range (ord ∘ f)
  /-
    case h.e'_3.h.e'_3
    ι : Sort u_4
    f : ι → Cardinal.{u_5}
    hf : BddAbove (Set.range f)
    ⊢ Eq (Set.range (Function.comp Cardinal.ord f)) (Set.image Cardinal.ord (Set.r …
  -/
  rw [range_comp]
  /-
    🎉 no goals
  -/


theorem lift_card_sInf_compl_le (s : Set Ordinal.{u}) :
    Cardinal.lift.{u + 1} (sInf sᶜ).card ≤ #s := by
  /-
    s : Set Ordinal.{u}
    ⊢ LE.le (Cardinal.lift.{u + 1, u} (InfSet.sInf (HasCompl.compl s)).card) (Card …
  -/
  rw [← mk_Iio_ordinal]
  /-
    s : Set Ordinal.{u}
    ⊢ LE.le (Cardinal.mk ↑(Set.Iio (InfSet.sInf (HasCompl.compl s)))) (Cardinal.mk …
  -/
  refine mk_le_mk_of_subset fun x (hx : x < _) ↦ ?_
  /-
    s : Set Ordinal.{u}
    x : Ordinal.{u}
    hx : LT.lt x (InfSet.sInf (HasCompl.compl s))
    ⊢ Membership.mem s x
  -/
  rw [← not_not_mem]
  /-
    s : Set Ordinal.{u}
    x : Ordinal.{u}
    hx : LT.lt x (InfSet.sInf (HasCompl.compl s))
    ⊢ Not (Not (Membership.mem s x))
  -/
  exact not_mem_of_lt_csInf' hx
  /-
    🎉 no goals
  -/


theorem card_sInf_range_compl_le_lift {ι : Type u} (f : ι → Ordinal.{max u v}) :
    (sInf (range f)ᶜ).card ≤ Cardinal.lift.{v} #ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (InfSet.sInf (HasCompl.compl (Set.range f))).card (Cardinal.lift.{v, u …
  -/
  rw [← Cardinal.lift_le.{max u v + 1}, Cardinal.lift_lift]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (Cardinal.lift.{(max u v) + 1, max u v} (InfSet.sInf (HasCompl.compl ( …
  -/
  apply (lift_card_sInf_compl_le _).trans
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (Cardinal.mk ↑(Set.range f)) (Cardinal.lift.{max v (u + 1) (v + 1), u} …
  -/
  rw [← Cardinal.lift_id'.{u, max u v + 1} #(range _)]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (Cardinal.lift.{u, max u ((max u v) + 1)} (Cardinal.mk ↑(Set.range f)) …
  -/
  exact mk_range_le_lift
  /-
    🎉 no goals
  -/


theorem card_sInf_range_compl_le {ι : Type u} (f : ι → Ordinal.{u}) :
    (sInf (range f)ᶜ).card ≤ #ι :=
  Cardinal.lift_id #ι ▸ card_sInf_range_compl_le_lift f


theorem sInf_compl_lt_lift_ord_succ {ι : Type u} (f : ι → Ordinal.{max u v}) :
    sInf (range f)ᶜ < lift.{v} (succ #ι).ord := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LT.lt (InfSet.sInf (HasCompl.compl (Set.range f))) (Ordinal.lift.{v, u} (Ord …
  -/
  rw [lift_ord, Cardinal.lift_succ, ← card_le_iff]
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ LE.le (InfSet.sInf (HasCompl.compl (Set.range f))).card (Cardinal.lift.{v, u …
  -/
  exact card_sInf_range_compl_le_lift f
  /-
    🎉 no goals
  -/


theorem sInf_compl_lt_ord_succ {ι : Type u} (f : ι → Ordinal.{u}) :
    sInf (range f)ᶜ < (succ #ι).ord :=
  lift_id (succ #ι).ord ▸ sInf_compl_lt_lift_ord_succ f

-- TODO: remove `bsup` in favor of `iSup` in a future refactor.


set_option linter.deprecated false in
private theorem sup_le_sup {ι ι' : Type u} (r : ι → ι → Prop) (r' : ι' → ι' → Prop)
    [IsWellOrder ι r] [IsWellOrder ι' r'] {o} (ho : type r = o) (ho' : type r' = o)
    (f : ∀ a < o, Ordinal.{max u v}) :
    sup.{_, v} (familyOfBFamily' r ho f) ≤ sup.{_, v} (familyOfBFamily' r' ho' f) :=
  sup_le fun i => by
    cases'
      typein_surj r'
        (by
          rw [ho', ← ho]
          exact typein_lt_type r i) with
      j hj
    /-
      case intro
      ι ι' : Type u
      r : ι → ι → Prop
      r' : ι' → ι' → Prop
      inst✝¹ : IsWellOrder ι r
      inst✝ : IsWellOrder ι' r'
      o : Ordinal.{u}
      ho : Eq (Ordinal.type r) o
      ho' : Eq (Ordinal.type r') o
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
      i : ι
      j : ι'
      hj : Eq ((Ordinal.typein r').toRelEmbedding j) ((Ordinal.typein r).toRelEmbedd …
      ⊢ LE.le (Ordinal.familyOfBFamily' r ho f i) (Ordinal.sup (Ordinal.familyOfBFam …
    -/
    simp_rw [familyOfBFamily', ← hj]
    /-
      case intro
      ι ι' : Type u
      r : ι → ι → Prop
      r' : ι' → ι' → Prop
      inst✝¹ : IsWellOrder ι r
      inst✝ : IsWellOrder ι' r'
      o : Ordinal.{u}
      ho : Eq (Ordinal.type r) o
      ho' : Eq (Ordinal.type r') o
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
      i : ι
      j : ι'
      hj : Eq ((Ordinal.typein r').toRelEmbedding j) ((Ordinal.typein r).toRelEmbedd …
      ⊢ LE.le (f ((Ordinal.typein r').toRelEmbedding j) ⋯) (Ordinal.sup (Ordinal.fam …
    -/
    apply le_sup
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
theorem sup_eq_sup {ι ι' : Type u} (r : ι → ι → Prop) (r' : ι' → ι' → Prop) [IsWellOrder ι r]
    [IsWellOrder ι' r'] {o : Ordinal.{u}} (ho : type r = o) (ho' : type r' = o)
    (f : ∀ a < o, Ordinal.{max u v}) :
    sup.{_, v} (familyOfBFamily' r ho f) = sup.{_, v} (familyOfBFamily' r' ho' f) :=
                                   /-
                                     ι ι' : Type u
                                     r : ι → ι → Prop
                                     r' : ι' → ι' → Prop
                                     inst✝¹ : IsWellOrder ι r
                                     inst✝ : IsWellOrder ι' r'
                                     o : Ordinal.{u}
                                     ho : Eq (Ordinal.type r) o
                                     ho' : Eq (Ordinal.type r') o
                                     f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
                                     ⊢ Eq (Set.range (Ordinal.familyOfBFamily' r ho f)) (Set.range (Ordinal.familyO …
                                   -/
  sup_eq_of_range_eq.{u, u, v} (by simp)
                                   /-
                                     🎉 no goals
                                   -/


set_option linter.deprecated false in
/-- The supremum of a family of ordinals indexed by the set of ordinals less than some
    `o : Ordinal.{u}`. This is a special case of `sup` over the family provided by
    `familyOfBFamily`. -/
def bsup (o : Ordinal.{u}) (f : ∀ a < o, Ordinal.{max u v}) : Ordinal.{max u v} :=
  sup.{_, v} (familyOfBFamily o f)


set_option linter.deprecated false in
@[simp]
theorem sup_eq_bsup {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    sup.{_, v} (familyOfBFamily o f) = bsup.{_, v} o f :=
  rfl


set_option linter.deprecated false in
@[simp]
theorem sup_eq_bsup' {o : Ordinal.{u}} {ι} (r : ι → ι → Prop) [IsWellOrder ι r] (ho : type r = o)
    (f : ∀ a < o, Ordinal.{max u v}) : sup.{_, v} (familyOfBFamily' r ho f) = bsup.{_, v} o f :=
  sup_eq_sup r _ ho _ f


theorem sSup_eq_bsup {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    sSup (brange o f) = bsup.{_, v} o f := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Eq (SupSet.sSup (o.brange f)) (o.bsup f)
  -/
  congr
  /-
    case e_a
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Eq (o.brange f) (Set.range (o.familyOfBFamily f))
  -/
  rw [range_familyOfBFamily]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[simp]
theorem bsup_eq_sup' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] (f : ι → Ordinal.{max u v}) :
    bsup.{_, v} _ (bfamilyOfFamily' r f) = sup.{_, v} f := by
  simp (config := { unfoldPartialApp := true }) only [← sup_eq_bsup' r, enum_typein,
    familyOfBFamily', bfamilyOfFamily']


theorem bsup_eq_bsup {ι : Type u} (r r' : ι → ι → Prop) [IsWellOrder ι r] [IsWellOrder ι r']
    (f : ι → Ordinal.{max u v}) :
    bsup.{_, v} _ (bfamilyOfFamily' r f) = bsup.{_, v} _ (bfamilyOfFamily' r' f) := by
  /-
    ι : Type u
    r r' : ι → ι → Prop
    inst✝¹ : IsWellOrder ι r
    inst✝ : IsWellOrder ι r'
    f : ι → Ordinal.{max u v}
    ⊢ Eq ((Ordinal.type r).bsup (Ordinal.bfamilyOfFamily' r f)) ((Ordinal.type r') …
  -/
  rw [bsup_eq_sup', bsup_eq_sup']
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[simp]
theorem bsup_eq_sup {ι : Type u} (f : ι → Ordinal.{max u v}) :
    bsup.{_, v} _ (bfamilyOfFamily f) = sup.{_, v} f :=
  bsup_eq_sup' _ f


@[congr]
theorem bsup_congr {o₁ o₂ : Ordinal.{u}} (f : ∀ a < o₁, Ordinal.{max u v}) (ho : o₁ = o₂) :
    bsup.{_, v} o₁ f = bsup.{_, v} o₂ fun a h => f a (h.trans_eq ho.symm) := by
  /-
    o₁ o₂ : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o₁ → Ordinal.{max u v}
    ho : Eq o₁ o₂
    ⊢ Eq (o₁.bsup f) (o₂.bsup fun a h => f a ⋯)
  -/
  subst ho
  -- Porting note: `rfl` is required.
  /-
    o₁ : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o₁ → Ordinal.{max u v}
    ⊢ Eq (o₁.bsup f) (o₁.bsup fun a h => f a ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
theorem bsup_le_iff {o f a} : bsup.{u, v} o f ≤ a ↔ ∀ i h, f i h ≤ a :=
  sup_le_iff.trans
    ⟨fun h i hi => by
      /-
        o : Ordinal.{u}
        f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
        a : Ordinal.{max u v}
        h : ∀ (i : o.toType), LE.le (o.familyOfBFamily f i) a
        i : Ordinal.{u}
        hi : LT.lt i o
        ⊢ LE.le (f i hi) a
      -/
      rw [← familyOfBFamily_enum o f]
      /-
        o : Ordinal.{u}
        f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
        a : Ordinal.{max u v}
        h : ∀ (i : o.toType), LE.le (o.familyOfBFamily f i) a
        i : Ordinal.{u}
        hi : LT.lt i o
        ⊢ LE.le (o.familyOfBFamily f ((Ordinal.enum fun x1 x2 => LT.lt x1 x2) ⟨i, ⋯⟩)) a
      -/
      exact h _, fun h _ => h _ _⟩
      /-
        🎉 no goals
      -/


theorem bsup_le {o : Ordinal} {f : ∀ b < o, Ordinal} {a} :
    (∀ i h, f i h ≤ a) → bsup.{u, v} o f ≤ a :=
  bsup_le_iff.2


theorem le_bsup {o} (f : ∀ a < o, Ordinal) (i h) : f i h ≤ bsup o f :=
  bsup_le_iff.1 le_rfl _ _


theorem lt_bsup {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) {a} :
    a < bsup.{_, v} o f ↔ ∃ i hi, a < f i hi := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    a : Ordinal.{max u v}
    ⊢ Iff (LT.lt a (o.bsup f)) (Exists fun i => Exists fun hi => LT.lt a (f i hi))
  -/
  simpa only [not_forall, not_le] using not_congr (@bsup_le_iff.{_, v} _ f a)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
theorem IsNormal.bsup {f : Ordinal.{max u v} → Ordinal.{max u w}} (H : IsNormal f)
    {o : Ordinal.{u}} :
    ∀ (g : ∀ a < o, Ordinal), o ≠ 0 → f (bsup.{_, v} o g) = bsup.{_, w} o fun a h => f (g a h) :=
  inductionOn o fun α r _ g h => by
    /-
      f : Ordinal.{max u v} → Ordinal.{max u w}
      H : Ordinal.IsNormal f
      o : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝ : IsWellOrder α r
      g : (a : Ordinal.{u}) → LT.lt a (Ordinal.type r) → Ordinal.{max u v}
      h : Ne (Ordinal.type r) 0
      ⊢ Eq (f ((Ordinal.type r).bsup g)) ((Ordinal.type r).bsup fun a h => f (g a h))
    -/
    haveI := type_ne_zero_iff_nonempty.1 h
    /-
      f : Ordinal.{max u v} → Ordinal.{max u w}
      H : Ordinal.IsNormal f
      o : Ordinal.{u}
      α : Type u
      r : α → α → Prop
      x✝ : IsWellOrder α r
      g : (a : Ordinal.{u}) → LT.lt a (Ordinal.type r) → Ordinal.{max u v}
      h : Ne (Ordinal.type r) 0
      this : Nonempty α
      ⊢ Eq (f ((Ordinal.type r).bsup g)) ((Ordinal.type r).bsup fun a h => f (g a h))
    -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    rw [← sup_eq_bsup' r, IsNormal.sup.{_, v, w} H, ← sup_eq_bsup' r] <;> rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


theorem lt_bsup_of_ne_bsup {o : Ordinal.{u}} {f : ∀ a < o, Ordinal.{max u v}} :
    (∀ i h, f i h ≠ bsup.{_, v} o f) ↔ ∀ i h, f i h < bsup.{_, v} o f :=
  ⟨fun hf _ _ => lt_of_le_of_ne (le_bsup _ _ _) (hf _ _), fun hf _ _ => ne_of_lt (hf _ _)⟩


set_option linter.deprecated false in
theorem bsup_not_succ_of_ne_bsup {o : Ordinal.{u}} {f : ∀ a < o, Ordinal.{max u v}}
    (hf : ∀ {i : Ordinal} (h : i < o), f i h ≠ bsup.{_, v} o f) (a) :
    a < bsup.{_, v} o f → succ a < bsup.{_, v} o f := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    hf : ∀ {i : Ordinal.{u}} (h : LT.lt i o), Ne (f i h) (o.bsup f)
    a : Ordinal.{max u v}
    ⊢ LT.lt a (o.bsup f) → LT.lt (Order.succ a) (o.bsup f)
  -/
  rw [← sup_eq_bsup] at *
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    hf : ∀ {i : Ordinal.{u}} (h : LT.lt i o), Ne (f i h) (Ordinal.sup (o.familyOfB …
    a : Ordinal.{max u v}
    ⊢ LT.lt a (Ordinal.sup (o.familyOfBFamily f)) → LT.lt (Order.succ a) (Ordinal. …
  -/
  exact sup_not_succ_of_ne_sup fun i => hf _
  /-
    🎉 no goals
  -/


@[simp]
theorem bsup_eq_zero_iff {o} {f : ∀ a < o, Ordinal} : bsup o f = 0 ↔ ∀ i hi, f i hi = 0 := by
  refine
    ⟨fun h i hi => ?_, fun h =>
      le_antisymm (bsup_le fun i hi => Ordinal.le_zero.2 (h i hi)) (Ordinal.zero_le _)⟩
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_5 u_4}
    h : Eq (o.bsup f) 0
    i : Ordinal.{u_4}
    hi : LT.lt i o
    ⊢ Eq (f i hi) 0
  -/
  rw [← Ordinal.le_zero, ← h]
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_5 u_4}
    h : Eq (o.bsup f) 0
    i : Ordinal.{u_4}
    hi : LT.lt i o
    ⊢ LE.le (f i hi) (o.bsup f)
  -/
  exact le_bsup f i hi
  /-
    🎉 no goals
  -/


theorem lt_bsup_of_limit {o : Ordinal} {f : ∀ a < o, Ordinal}
    (hf : ∀ {a a'} (ha : a < o) (ha' : a' < o), a < a' → f a ha < f a' ha')
    (ho : ∀ a < o, succ a < o) (i h) : f i h < bsup o f :=
  (hf _ _ <| lt_succ i).trans_le (le_bsup f (succ i) <| ho _ h)


theorem bsup_succ_of_mono {o : Ordinal} {f : ∀ a < succ o, Ordinal}
    (hf : ∀ {i j} (hi hj), i ≤ j → f i hi ≤ f j hj) : bsup _ f = f o (lt_succ o) :=
  le_antisymm (bsup_le fun _i hi => hf _ _ <| le_of_lt_succ hi) (le_bsup _ _ _)


@[simp]
theorem bsup_zero (f : ∀ a < (0 : Ordinal), Ordinal) : bsup 0 f = 0 :=
  bsup_eq_zero_iff.2 fun i hi => (Ordinal.not_lt_zero i hi).elim


theorem bsup_const {o : Ordinal.{u}} (ho : o ≠ 0) (a : Ordinal.{max u v}) :
    (bsup.{_, v} o fun _ _ => a) = a :=
  le_antisymm (bsup_le fun _ _ => le_rfl) (le_bsup _ 0 (Ordinal.pos_iff_ne_zero.2 ho))


set_option linter.deprecated false in
@[simp]
theorem bsup_one (f : ∀ a < (1 : Ordinal), Ordinal) : bsup 1 f = f 0 zero_lt_one := by
  /-
    f : (a : Ordinal.{u_4}) → LT.lt a 1 → Ordinal.{max u_4 u_5}
    ⊢ Eq (Ordinal.bsup 1 f) (f 0 ⋯)
  -/
  simp_rw [← sup_eq_bsup, sup_unique, familyOfBFamily, familyOfBFamily', typein_one_toType]
  /-
    🎉 no goals
  -/


theorem bsup_le_of_brange_subset {o o'} {f : ∀ a < o, Ordinal} {g : ∀ a < o', Ordinal}
    (h : brange o f ⊆ brange o' g) : bsup.{u, max v w} o f ≤ bsup.{v, max u w} o' g :=
  bsup_le fun i hi => by
    /-
      o : Ordinal.{u}
      o' : Ordinal.{v}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max (max u v) w}
      g : (a : Ordinal.{v}) → LT.lt a o' → Ordinal.{max (max u v) w}
      h : HasSubset.Subset (o.brange f) (o'.brange g)
      i : Ordinal.{u}
      hi : LT.lt i o
      ⊢ LE.le (f i hi) (o'.bsup g)
    -/
    obtain ⟨j, hj, hj'⟩ := h ⟨i, hi, rfl⟩
    /-
      case intro.intro
      o : Ordinal.{u}
      o' : Ordinal.{v}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max (max u v) w}
      g : (a : Ordinal.{v}) → LT.lt a o' → Ordinal.{max (max u v) w}
      h : HasSubset.Subset (o.brange f) (o'.brange g)
      i : Ordinal.{u}
      hi : LT.lt i o
      j : Ordinal.{v}
      hj : LT.lt j o'
      hj' : Eq (g j hj) (f i hi)
      ⊢ LE.le (f i hi) (o'.bsup g)
    -/
    rw [← hj']
    /-
      case intro.intro
      o : Ordinal.{u}
      o' : Ordinal.{v}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max (max u v) w}
      g : (a : Ordinal.{v}) → LT.lt a o' → Ordinal.{max (max u v) w}
      h : HasSubset.Subset (o.brange f) (o'.brange g)
      i : Ordinal.{u}
      hi : LT.lt i o
      j : Ordinal.{v}
      hj : LT.lt j o'
      hj' : Eq (g j hj) (f i hi)
      ⊢ LE.le (g j hj) (o'.bsup g)
    -/
    apply le_bsup
    /-
      🎉 no goals
    -/


theorem bsup_eq_of_brange_eq {o o'} {f : ∀ a < o, Ordinal} {g : ∀ a < o', Ordinal}
    (h : brange o f = brange o' g) : bsup.{u, max v w} o f = bsup.{v, max u w} o' g :=
  (bsup_le_of_brange_subset.{u, v, w} h.le).antisymm (bsup_le_of_brange_subset.{v, u, w} h.ge)


set_option linter.deprecated false in
theorem iSup_eq_bsup {o} {f : ∀ a < o, Ordinal} : ⨆ a : Iio o, f a.1 a.2 = bsup o f := by
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_5 u_4}
    ⊢ Eq (iSup fun a => f ↑a ⋯) (o.bsup f)
  -/
  simp_rw [Iio, bsup, sup, iSup, range_familyOfBFamily, brange, range, Subtype.exists, mem_setOf]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- The least strict upper bound of a family of ordinals. -/
def lsub {ι} (f : ι → Ordinal) : Ordinal :=
  sup (succ ∘ f)


set_option linter.deprecated false in
@[simp]
theorem sup_eq_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) :
    sup.{_, v} (succ ∘ f) = lsub.{_, v} f :=
  rfl


set_option linter.deprecated false in
theorem lsub_le_iff {ι : Type u} {f : ι → Ordinal.{max u v}} {a} :
    lsub.{_, v} f ≤ a ↔ ∀ i, f i < a := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    a : Ordinal.{max v u}
    ⊢ Iff (LE.le (Ordinal.lsub f) a) (∀ (i : ι), LT.lt (f i) a)
  -/
  convert sup_le_iff.{_, v} (f := succ ∘ f) (a := a) using 2
  -- Porting note: `comp_apply` is required.
  /-
    case h.e'_2.h.a
    ι : Type u
    f : ι → Ordinal.{max u v}
    a : Ordinal.{max v u}
    a✝ : ι
    ⊢ Iff (LT.lt (f a✝) a) (LE.le (Function.comp Order.succ f a✝) a)
  -/
  simp only [comp_apply, succ_le_iff]
  /-
    🎉 no goals
  -/


theorem lsub_le {ι} {f : ι → Ordinal} {a} : (∀ i, f i < a) → lsub f ≤ a :=
  lsub_le_iff.2


set_option linter.deprecated false in
theorem lt_lsub {ι} (f : ι → Ordinal) (i) : f i < lsub f :=
  succ_le_iff.1 (le_sup _ i)


theorem lt_lsub_iff {ι : Type u} {f : ι → Ordinal.{max u v}} {a} :
    a < lsub.{_, v} f ↔ ∃ i, a ≤ f i := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    a : Ordinal.{max v u}
    ⊢ Iff (LT.lt a (Ordinal.lsub f)) (Exists fun i => LE.le a (f i))
  -/
  simpa only [not_forall, not_lt, not_le] using not_congr (@lsub_le_iff.{_, v} _ f a)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
theorem sup_le_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) : sup.{_, v} f ≤ lsub.{_, v} f :=
  sup_le fun i => (lt_lsub f i).le


set_option linter.deprecated false in
theorem lsub_le_sup_succ {ι : Type u} (f : ι → Ordinal.{max u v}) :
    lsub.{_, v} f ≤ succ (sup.{_, v} f) :=
  lsub_le fun i => lt_succ_iff.2 (le_sup f i)


set_option linter.deprecated false in
theorem sup_eq_lsub_or_sup_succ_eq_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) :
    sup.{_, v} f = lsub.{_, v} f ∨ succ (sup.{_, v} f) = lsub.{_, v} f := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ Or (Eq (Ordinal.sup f) (Ordinal.lsub f)) (Eq (Order.succ (Ordinal.sup f)) (O …
  -/
  cases' eq_or_lt_of_le (sup_le_lsub.{_, v} f) with h h
    /-
      case inl
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : Eq (Ordinal.sup f) (Ordinal.lsub f)
      ⊢ Or (Eq (Ordinal.sup f) (Ordinal.lsub f)) (Eq (Order.succ (Ordinal.sup f)) (O …
    -/
  · exact Or.inl h
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : LT.lt (Ordinal.sup f) (Ordinal.lsub f)
      ⊢ Or (Eq (Ordinal.sup f) (Ordinal.lsub f)) (Eq (Order.succ (Ordinal.sup f)) (O …
    -/
  · exact Or.inr ((succ_le_of_lt h).antisymm (lsub_le_sup_succ f))
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
theorem sup_succ_le_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) :
    succ (sup.{_, v} f) ≤ lsub.{_, v} f ↔ ∃ i, f i = sup.{_, v} f := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ Iff (LE.le (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)) (Exists fun i => E …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : LE.le (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)
      ⊢ Exists fun i => Eq (f i) (Ordinal.sup f)
    -/
  · by_contra! hf
    /-
      case refine_1
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : LE.le (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)
      hf : ∀ (i : ι), Ne (f i) (Ordinal.sup f)
      ⊢ False
    -/
    exact (succ_le_iff.1 h).ne ((sup_le_lsub f).antisymm (lsub_le (ne_sup_iff_lt_sup.1 hf)))
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ (Exists fun i => Eq (f i) (Ordinal.sup f)) → LE.le (Order.succ (Ordinal.sup  …
  -/
  rintro ⟨_, hf⟩
  /-
    case refine_2.intro
    ι : Type u
    f : ι → Ordinal.{max u v}
    w✝ : ι
    hf : Eq (f w✝) (Ordinal.sup f)
    ⊢ LE.le (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)
  -/
  rw [succ_le_iff, ← hf]
  /-
    case refine_2.intro
    ι : Type u
    f : ι → Ordinal.{max u v}
    w✝ : ι
    hf : Eq (f w✝) (Ordinal.sup f)
    ⊢ LT.lt (f w✝) (Ordinal.lsub f)
  -/
  exact lt_lsub _ _
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
theorem sup_succ_eq_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) :
    succ (sup.{_, v} f) = lsub.{_, v} f ↔ ∃ i, f i = sup.{_, v} f :=
  (lsub_le_sup_succ f).le_iff_eq.symm.trans (sup_succ_le_lsub f)


set_option linter.deprecated false in
theorem sup_eq_lsub_iff_succ {ι : Type u} (f : ι → Ordinal.{max u v}) :
    sup.{_, v} f = lsub.{_, v} f ↔ ∀ a < lsub.{_, v} f, succ a < lsub.{_, v} f := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ Iff (Eq (Ordinal.sup f) (Ordinal.lsub f)) (∀ (a : Ordinal.{max v u}), LT.lt  …
  -/
  refine ⟨fun h => ?_, fun hf => le_antisymm (sup_le_lsub f) (lsub_le fun i => ?_)⟩
    /-
      case refine_1
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : Eq (Ordinal.sup f) (Ordinal.lsub f)
      ⊢ ∀ (a : Ordinal.{max v u}), LT.lt a (Ordinal.lsub f) → LT.lt (Order.succ a) ( …
    -/
  · rw [← h]
    /-
      case refine_1
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : Eq (Ordinal.sup f) (Ordinal.lsub f)
      ⊢ ∀ (a : Ordinal.{max v u}), LT.lt a (Ordinal.sup f) → LT.lt (Order.succ a) (O …
    -/
    exact fun a => sup_not_succ_of_ne_sup fun i => (lsub_le_iff.1 (le_of_eq h.symm) i).ne
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    ι : Type u
    f : ι → Ordinal.{max u v}
    hf : ∀ (a : Ordinal.{max v u}), LT.lt a (Ordinal.lsub f) → LT.lt (Order.succ a …
    i : ι
    ⊢ LT.lt (f i) (Ordinal.sup f)
  -/
  by_contra! hle
  /-
    case refine_2
    ι : Type u
    f : ι → Ordinal.{max u v}
    hf : ∀ (a : Ordinal.{max v u}), LT.lt a (Ordinal.lsub f) → LT.lt (Order.succ a …
    i : ι
    hle : LE.le (Ordinal.sup f) (f i)
    ⊢ False
  -/
  have heq := (sup_succ_eq_lsub f).2 ⟨i, le_antisymm (le_sup _ _) hle⟩
  have :=
    hf _
      (by
        rw [← heq]
        exact lt_succ (sup f))
  /-
    case refine_2
    ι : Type u
    f : ι → Ordinal.{max u v}
    hf : ∀ (a : Ordinal.{max v u}), LT.lt a (Ordinal.lsub f) → LT.lt (Order.succ a …
    i : ι
    hle : LE.le (Ordinal.sup f) (f i)
    heq : Eq (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)
    this : LT.lt (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)
    ⊢ False
  -/
  rw [heq] at this
  /-
    case refine_2
    ι : Type u
    f : ι → Ordinal.{max u v}
    hf : ∀ (a : Ordinal.{max v u}), LT.lt a (Ordinal.lsub f) → LT.lt (Order.succ a …
    i : ι
    hle : LE.le (Ordinal.sup f) (f i)
    heq : Eq (Order.succ (Ordinal.sup f)) (Ordinal.lsub f)
    this : LT.lt (Ordinal.lsub f) (Ordinal.lsub f)
    ⊢ False
  -/
  exact this.false
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
theorem sup_eq_lsub_iff_lt_sup {ι : Type u} (f : ι → Ordinal.{max u v}) :
    sup.{_, v} f = lsub.{_, v} f ↔ ∀ i, f i < sup.{_, v} f :=
  ⟨fun h i => by
    /-
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : Eq (Ordinal.sup f) (Ordinal.lsub f)
      i : ι
      ⊢ LT.lt (f i) (Ordinal.sup f)
    -/
    rw [h]
    /-
      ι : Type u
      f : ι → Ordinal.{max u v}
      h : Eq (Ordinal.sup f) (Ordinal.lsub f)
      i : ι
      ⊢ LT.lt (f i) (Ordinal.lsub f)
    -/
    apply lt_lsub, fun h => le_antisymm (sup_le_lsub f) (lsub_le h)⟩
    /-
      🎉 no goals
    -/


@[simp]
theorem lsub_empty {ι} [h : IsEmpty ι] (f : ι → Ordinal) : lsub f = 0 := by
  /-
    ι : Type u_4
    h : IsEmpty ι
    f : ι → Ordinal.{max u_5 u_4}
    ⊢ Eq (Ordinal.lsub f) 0
  -/
  rw [← Ordinal.le_zero, lsub_le_iff]
  /-
    ι : Type u_4
    h : IsEmpty ι
    f : ι → Ordinal.{max u_5 u_4}
    ⊢ ∀ (i : ι), LT.lt (f i) 0
  -/
  exact h.elim
  /-
    🎉 no goals
  -/


theorem lsub_pos {ι : Type u} [h : Nonempty ι] (f : ι → Ordinal.{max u v}) : 0 < lsub.{_, v} f :=
  h.elim fun i => (Ordinal.zero_le _).trans_lt (lt_lsub f i)


@[simp]
theorem lsub_eq_zero_iff {ι : Type u} (f : ι → Ordinal.{max u v}) :
    lsub.{_, v} f = 0 ↔ IsEmpty ι := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ Iff (Eq (Ordinal.lsub f) 0) (IsEmpty ι)
  -/
  refine ⟨fun h => ⟨fun i => ?_⟩, fun h => @lsub_empty _ h _⟩
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    h : Eq (Ordinal.lsub f) 0
    i : ι
    ⊢ False
  -/
  have := @lsub_pos.{_, v} _ ⟨i⟩ f
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    h : Eq (Ordinal.lsub f) 0
    i : ι
    this : LT.lt 0 (Ordinal.lsub f)
    ⊢ False
  -/
  rw [h] at this
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    h : Eq (Ordinal.lsub f) 0
    i : ι
    this : LT.lt 0 0
    ⊢ False
  -/
  exact this.false
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[simp]
theorem lsub_const {ι} [Nonempty ι] (o : Ordinal) : (lsub fun _ : ι => o) = succ o :=
  sup_const (succ o)


set_option linter.deprecated false in
@[simp]
theorem lsub_unique {ι} [Unique ι] (f : ι → Ordinal) : lsub f = succ (f default) :=
  sup_unique _


set_option linter.deprecated false in
theorem lsub_le_of_range_subset {ι ι'} {f : ι → Ordinal} {g : ι' → Ordinal}
    (h : Set.range f ⊆ Set.range g) : lsub.{u, max v w} f ≤ lsub.{v, max u w} g :=
                                       /-
                                         ι : Type u
                                         ι' : Type v
                                         f : ι → Ordinal.{max (max u v) w}
                                         g : ι' → Ordinal.{max (max u v) w}
                                         h : HasSubset.Subset (Set.range f) (Set.range g)
                                         ⊢ HasSubset.Subset (Set.range (Function.comp Order.succ f)) (Set.range (Functi …
                                       -/
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  sup_le_of_range_subset.{u, v, w} (by convert Set.image_subset succ h <;> apply Set.range_comp)
                                                                           /-
                                                                             🎉 no goals
                                                                           -/


theorem lsub_eq_of_range_eq {ι ι'} {f : ι → Ordinal} {g : ι' → Ordinal}
    (h : Set.range f = Set.range g) : lsub.{u, max v w} f = lsub.{v, max u w} g :=
  (lsub_le_of_range_subset.{u, v, w} h.le).antisymm (lsub_le_of_range_subset.{v, u, w} h.ge)


set_option linter.deprecated false in
@[simp]
theorem lsub_sum {α : Type u} {β : Type v} (f : α ⊕ β → Ordinal) :
    lsub.{max u v, w} f =
      max (lsub.{u, max v w} fun a => f (Sum.inl a)) (lsub.{v, max u w} fun b => f (Sum.inr b)) :=
  sup_sum _


theorem lsub_not_mem_range {ι : Type u} (f : ι → Ordinal.{max u v}) :
    lsub.{_, v} f ∉ Set.range f := fun ⟨i, h⟩ =>
  h.not_lt (lt_lsub f i)


theorem nonempty_compl_range {ι : Type u} (f : ι → Ordinal.{max u v}) : (Set.range f)ᶜ.Nonempty :=
  ⟨_, lsub_not_mem_range.{_, v} f⟩


set_option linter.deprecated false in
@[simp]
theorem lsub_typein (o : Ordinal) : lsub.{u, u} (typein (α := o.toType) (· < ·)) = o :=
  (lsub_le.{u, u} typein_lt_self).antisymm
    (by
      /-
        o : Ordinal.{u}
        ⊢ LE.le o (Ordinal.lsub ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedd …
      -/
      by_contra! h
      -- Porting note: `nth_rw` → `conv_rhs` & `rw`
      /-
        o : Ordinal.{u}
        h : LT.lt (Ordinal.lsub ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedd …
        ⊢ False
      -/
      conv_rhs at h => rw [← type_lt o]
      /-
        o : Ordinal.{u}
        h : LT.lt (Ordinal.lsub ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedd …
        ⊢ False
      -/
      simpa [typein_enum] using lt_lsub.{u, u} (typein (· < ·)) (enum (· < ·) ⟨_, h⟩))
      /-
        🎉 no goals
      -/


set_option linter.deprecated false in
theorem sup_typein_limit {o : Ordinal} (ho : ∀ a, a < o → succ a < o) :
    sup.{u, u} (typein ((· < ·) : o.toType → o.toType → Prop)) = o := by
  -- Porting note: `rwa` → `rw` & `assumption`
  /-
    o : Ordinal.{u}
    ho : ∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (Order.succ a) o
    ⊢ Eq (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding) o
  -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  rw [(sup_eq_lsub_iff_succ.{u, u} (typein (· < ·))).2] <;> rw [lsub_typein o]; assumption
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


set_option linter.deprecated false in
@[simp]
theorem sup_typein_succ {o : Ordinal} :
    sup.{u, u} (typein ((· < ·) : (succ o).toType → (succ o).toType → Prop)) = o := by
  cases'
    sup_eq_lsub_or_sup_succ_eq_lsub.{u, u}
      (typein ((· < ·) : (succ o).toType → (succ o).toType → Prop)) with
    h h
    /-
      case inl
      o : Ordinal.{u}
      h : Eq (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding) …
      ⊢ Eq (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding) o
    -/
  · rw [sup_eq_lsub_iff_succ] at h
    /-
      case inl
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt a (Ordinal.lsub ⇑(Ordinal.typein fun x1 x2 => L …
      ⊢ Eq (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding) o
    -/
    simp only [lsub_typein] at h
    /-
      case inl
      o : Ordinal.{u}
      h : ∀ (a : Ordinal.{u}), LT.lt a (Order.succ o) → LT.lt (Order.succ a) (Order. …
      ⊢ Eq (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding) o
    -/
    exact (h o (lt_succ o)).false.elim
    /-
      🎉 no goals
    -/
  /-
    case inr
    o : Ordinal.{u}
    h : Eq (Order.succ (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toR …
    ⊢ Eq (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding) o
  -/
  rw [← succ_eq_succ_iff, h]
  /-
    case inr
    o : Ordinal.{u}
    h : Eq (Order.succ (Ordinal.sup ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toR …
    ⊢ Eq (Ordinal.lsub ⇑(Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding)  …
  -/
  apply lsub_typein
  /-
    🎉 no goals
  -/


/-- The least strict upper bound of a family of ordinals indexed by the set of ordinals less than
    some `o : Ordinal.{u}`.

    This is to `lsub` as `bsup` is to `sup`. -/
def blsub (o : Ordinal.{u}) (f : ∀ a < o, Ordinal.{max u v}) : Ordinal.{max u v} :=
  bsup.{_, v} o fun a ha => succ (f a ha)


@[simp]
theorem bsup_eq_blsub (o : Ordinal.{u}) (f : ∀ a < o, Ordinal.{max u v}) :
    (bsup.{_, v} o fun a ha => succ (f a ha)) = blsub.{_, v} o f :=
  rfl


theorem lsub_eq_blsub' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r] {o} (ho : type r = o)
    (f : ∀ a < o, Ordinal.{max u v}) : lsub.{_, v} (familyOfBFamily' r ho f) = blsub.{_, v} o f :=
  sup_eq_bsup'.{_, v} r ho fun a ha => succ (f a ha)


theorem lsub_eq_lsub {ι ι' : Type u} (r : ι → ι → Prop) (r' : ι' → ι' → Prop) [IsWellOrder ι r]
    [IsWellOrder ι' r'] {o} (ho : type r = o) (ho' : type r' = o)
    (f : ∀ a < o, Ordinal.{max u v}) :
    lsub.{_, v} (familyOfBFamily' r ho f) = lsub.{_, v} (familyOfBFamily' r' ho' f) := by
  /-
    ι ι' : Type u
    r : ι → ι → Prop
    r' : ι' → ι' → Prop
    inst✝¹ : IsWellOrder ι r
    inst✝ : IsWellOrder ι' r'
    o : Ordinal.{u}
    ho : Eq (Ordinal.type r) o
    ho' : Eq (Ordinal.type r') o
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Eq (Ordinal.lsub (Ordinal.familyOfBFamily' r ho f)) (Ordinal.lsub (Ordinal.f …
  -/
  rw [lsub_eq_blsub', lsub_eq_blsub']
  /-
    🎉 no goals
  -/


@[simp]
theorem lsub_eq_blsub {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    lsub.{_, v} (familyOfBFamily o f) = blsub.{_, v} o f :=
  lsub_eq_blsub' _ _ _


@[simp]
theorem blsub_eq_lsub' {ι : Type u} (r : ι → ι → Prop) [IsWellOrder ι r]
    (f : ι → Ordinal.{max u v}) : blsub.{_, v} _ (bfamilyOfFamily' r f) = lsub.{_, v} f :=
  bsup_eq_sup'.{_, v} r (succ ∘ f)


theorem blsub_eq_blsub {ι : Type u} (r r' : ι → ι → Prop) [IsWellOrder ι r] [IsWellOrder ι r']
    (f : ι → Ordinal.{max u v}) :
    blsub.{_, v} _ (bfamilyOfFamily' r f) = blsub.{_, v} _ (bfamilyOfFamily' r' f) := by
  /-
    ι : Type u
    r r' : ι → ι → Prop
    inst✝¹ : IsWellOrder ι r
    inst✝ : IsWellOrder ι r'
    f : ι → Ordinal.{max u v}
    ⊢ Eq ((Ordinal.type r).blsub (Ordinal.bfamilyOfFamily' r f)) ((Ordinal.type r' …
  -/
  rw [blsub_eq_lsub', blsub_eq_lsub']
  /-
    🎉 no goals
  -/


@[simp]
theorem blsub_eq_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) :
    blsub.{_, v} _ (bfamilyOfFamily f) = lsub.{_, v} f :=
  blsub_eq_lsub' _ _


@[congr]
theorem blsub_congr {o₁ o₂ : Ordinal.{u}} (f : ∀ a < o₁, Ordinal.{max u v}) (ho : o₁ = o₂) :
    blsub.{_, v} o₁ f = blsub.{_, v} o₂ fun a h => f a (h.trans_eq ho.symm) := by
  /-
    o₁ o₂ : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o₁ → Ordinal.{max u v}
    ho : Eq o₁ o₂
    ⊢ Eq (o₁.blsub f) (o₂.blsub fun a h => f a ⋯)
  -/
  subst ho
  -- Porting note: `rfl` is required.
  /-
    o₁ : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o₁ → Ordinal.{max u v}
    ⊢ Eq (o₁.blsub f) (o₁.blsub fun a h => f a ⋯)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem blsub_le_iff {o : Ordinal.{u}} {f : ∀ a < o, Ordinal.{max u v}} {a} :
    blsub.{_, v} o f ≤ a ↔ ∀ i h, f i h < a := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    a : Ordinal.{max u v}
    ⊢ Iff (LE.le (o.blsub f) a) (∀ (i : Ordinal.{u}) (h : LT.lt i o), LT.lt (f i h …
  -/
  convert bsup_le_iff.{_, v} (f := fun a ha => succ (f a ha)) (a := a) using 2
  /-
    case h.e'_2.h.a
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    a : Ordinal.{max u v}
    a✝ : Ordinal.{u}
    ⊢ Iff (∀ (h : LT.lt a✝ o), LT.lt (f a✝ h) a) (∀ (h : LT.lt a✝ o), LE.le (Order …
  -/
  simp_rw [succ_le_iff]
  /-
    🎉 no goals
  -/


theorem blsub_le {o : Ordinal} {f : ∀ b < o, Ordinal} {a} : (∀ i h, f i h < a) → blsub o f ≤ a :=
  blsub_le_iff.2


theorem lt_blsub {o} (f : ∀ a < o, Ordinal) (i h) : f i h < blsub o f :=
  blsub_le_iff.1 le_rfl _ _


theorem lt_blsub_iff {o : Ordinal.{u}} {f : ∀ b < o, Ordinal.{max u v}} {a} :
    a < blsub.{_, v} o f ↔ ∃ i hi, a ≤ f i hi := by
  /-
    o : Ordinal.{u}
    f : (b : Ordinal.{u}) → LT.lt b o → Ordinal.{max u v}
    a : Ordinal.{max u v}
    ⊢ Iff (LT.lt a (o.blsub f)) (Exists fun i => Exists fun hi => LE.le a (f i hi))
  -/
  simpa only [not_forall, not_lt, not_le] using not_congr (@blsub_le_iff.{_, v} _ f a)
  /-
    🎉 no goals
  -/


theorem bsup_le_blsub {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    bsup.{_, v} o f ≤ blsub.{_, v} o f :=
  bsup_le fun i h => (lt_blsub f i h).le


theorem blsub_le_bsup_succ {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    blsub.{_, v} o f ≤ succ (bsup.{_, v} o f) :=
  blsub_le fun i h => lt_succ_iff.2 (le_bsup f i h)


theorem bsup_eq_blsub_or_succ_bsup_eq_blsub {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    bsup.{_, v} o f = blsub.{_, v} o f ∨ succ (bsup.{_, v} o f) = blsub.{_, v} o f := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Or (Eq (o.bsup f) (o.blsub f)) (Eq (Order.succ (o.bsup f)) (o.blsub f))
  -/
  rw [← sup_eq_bsup, ← lsub_eq_blsub]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Or (Eq (Ordinal.sup (o.familyOfBFamily f)) (Ordinal.lsub (o.familyOfBFamily  …
  -/
  exact sup_eq_lsub_or_sup_succ_eq_lsub _
  /-
    🎉 no goals
  -/


theorem bsup_succ_le_blsub {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    succ (bsup.{_, v} o f) ≤ blsub.{_, v} o f ↔ ∃ i hi, f i hi = bsup.{_, v} o f := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Iff (LE.le (Order.succ (o.bsup f)) (o.blsub f)) (Exists fun i => Exists fun  …
  -/
  refine ⟨fun h => ?_, ?_⟩
    /-
      case refine_1
      o : Ordinal.{u}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
      h : LE.le (Order.succ (o.bsup f)) (o.blsub f)
      ⊢ Exists fun i => Exists fun hi => Eq (f i hi) (o.bsup f)
    -/
  · by_contra! hf
    exact
      ne_of_lt (succ_le_iff.1 h)
        (le_antisymm (bsup_le_blsub f) (blsub_le (lt_bsup_of_ne_bsup.1 hf)))
  /-
    case refine_2
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ (Exists fun i => Exists fun hi => Eq (f i hi) (o.bsup f)) → LE.le (Order.suc …
  -/
  rintro ⟨_, _, hf⟩
  /-
    case refine_2.intro.intro
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    w✝¹ : Ordinal.{u}
    w✝ : LT.lt w✝¹ o
    hf : Eq (f w✝¹ w✝) (o.bsup f)
    ⊢ LE.le (Order.succ (o.bsup f)) (o.blsub f)
  -/
  rw [succ_le_iff, ← hf]
  /-
    case refine_2.intro.intro
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    w✝¹ : Ordinal.{u}
    w✝ : LT.lt w✝¹ o
    hf : Eq (f w✝¹ w✝) (o.bsup f)
    ⊢ LT.lt (f w✝¹ w✝) (o.blsub f)
  -/
  exact lt_blsub _ _ _
  /-
    🎉 no goals
  -/


theorem bsup_succ_eq_blsub {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    succ (bsup.{_, v} o f) = blsub.{_, v} o f ↔ ∃ i hi, f i hi = bsup.{_, v} o f :=
  (blsub_le_bsup_succ f).le_iff_eq.symm.trans (bsup_succ_le_blsub f)


theorem bsup_eq_blsub_iff_succ {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    bsup.{_, v} o f = blsub.{_, v} o f ↔ ∀ a < blsub.{_, v} o f, succ a < blsub.{_, v} o f := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Iff (Eq (o.bsup f) (o.blsub f)) (∀ (a : Ordinal.{max u v}), LT.lt a (o.blsub …
  -/
  rw [← sup_eq_bsup, ← lsub_eq_blsub]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    ⊢ Iff (Eq (Ordinal.sup (o.familyOfBFamily f)) (Ordinal.lsub (o.familyOfBFamily …
  -/
  apply sup_eq_lsub_iff_succ
  /-
    🎉 no goals
  -/


theorem bsup_eq_blsub_iff_lt_bsup {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    bsup.{_, v} o f = blsub.{_, v} o f ↔ ∀ i hi, f i hi < bsup.{_, v} o f :=
  ⟨fun h i => by
    /-
      o : Ordinal.{u}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
      h : Eq (o.bsup f) (o.blsub f)
      i : Ordinal.{u}
      ⊢ ∀ (hi : LT.lt i o), LT.lt (f i hi) (o.bsup f)
    -/
    rw [h]
    /-
      o : Ordinal.{u}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
      h : Eq (o.bsup f) (o.blsub f)
      i : Ordinal.{u}
      ⊢ ∀ (hi : LT.lt i o), LT.lt (f i hi) (o.blsub f)
    -/
    apply lt_blsub, fun h => le_antisymm (bsup_le_blsub f) (blsub_le h)⟩
    /-
      🎉 no goals
    -/


theorem bsup_eq_blsub_of_lt_succ_limit {o : Ordinal.{u}} (ho : IsLimit o)
    {f : ∀ a < o, Ordinal.{max u v}} (hf : ∀ a ha, f a ha < f (succ a) (ho.succ_lt ha)) :
    bsup.{_, v} o f = blsub.{_, v} o f := by
  /-
    o : Ordinal.{u}
    ho : o.IsLimit
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    hf : ∀ (a : Ordinal.{u}) (ha : LT.lt a o), LT.lt (f a ha) (f (Order.succ a) ⋯)
    ⊢ Eq (o.bsup f) (o.blsub f)
  -/
  rw [bsup_eq_blsub_iff_lt_bsup]
  /-
    o : Ordinal.{u}
    ho : o.IsLimit
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    hf : ∀ (a : Ordinal.{u}) (ha : LT.lt a o), LT.lt (f a ha) (f (Order.succ a) ⋯)
    ⊢ ∀ (i : Ordinal.{u}) (hi : LT.lt i o), LT.lt (f i hi) (o.bsup f)
  -/
  exact fun i hi => (hf i hi).trans_le (le_bsup f _ _)
  /-
    🎉 no goals
  -/


theorem blsub_succ_of_mono {o : Ordinal.{u}} {f : ∀ a < succ o, Ordinal.{max u v}}
    (hf : ∀ {i j} (hi hj), i ≤ j → f i hi ≤ f j hj) : blsub.{_, v} _ f = succ (f o (lt_succ o)) :=
  bsup_succ_of_mono fun {_ _} hi hj h => succ_le_succ (hf hi hj h)


@[simp]
theorem blsub_eq_zero_iff {o} {f : ∀ a < o, Ordinal} : blsub o f = 0 ↔ o = 0 := by
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_5 u_4}
    ⊢ Iff (Eq (o.blsub f) 0) (Eq o 0)
  -/
  rw [← lsub_eq_blsub, lsub_eq_zero_iff]
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_5 u_4}
    ⊢ Iff (IsEmpty o.toType) (Eq o 0)
  -/
  exact toType_empty_iff_eq_zero
  /-
    🎉 no goals
  -/

-- Porting note: `rwa` → `rw`

@[simp]
                                                                            /-
                                                                              f : (a : Ordinal.{u_4}) → LT.lt a 0 → Ordinal.{max u_4 u_5}
                                                                              ⊢ Eq (Ordinal.blsub 0 f) 0
                                                                            -/
theorem blsub_zero (f : ∀ a < (0 : Ordinal), Ordinal) : blsub 0 f = 0 := by rw [blsub_eq_zero_iff]
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem blsub_pos {o : Ordinal} (ho : 0 < o) (f : ∀ a < o, Ordinal) : 0 < blsub o f :=
  (Ordinal.zero_le _).trans_lt (lt_blsub f 0 ho)


theorem blsub_type {α : Type u} (r : α → α → Prop) [IsWellOrder α r]
    (f : ∀ a < type r, Ordinal.{max u v}) :
    blsub.{_, v} (type r) f = lsub.{_, v} fun a => f (typein r a) (typein_lt_type _ _) :=
  eq_of_forall_ge_iff fun o => by
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      f : (a : Ordinal.{u}) → LT.lt a (Ordinal.type r) → Ordinal.{max u v}
      o : Ordinal.{max u v}
      ⊢ Iff (LE.le ((Ordinal.type r).blsub f) o) (LE.le (Ordinal.lsub fun a => f ((O …
    -/
    rw [blsub_le_iff, lsub_le_iff]
    /-
      α : Type u
      r : α → α → Prop
      inst✝ : IsWellOrder α r
      f : (a : Ordinal.{u}) → LT.lt a (Ordinal.type r) → Ordinal.{max u v}
      o : Ordinal.{max u v}
      ⊢ Iff (∀ (i : Ordinal.{u}) (h : LT.lt i (Ordinal.type r)), LT.lt (f i h) o) (∀ …
    -/
    exact ⟨fun H b => H _ _, fun H i h => by simpa only [typein_enum] using H (enum r ⟨i, h⟩)⟩
    /-
      🎉 no goals
    -/


theorem blsub_const {o : Ordinal} (ho : o ≠ 0) (a : Ordinal) :
    (blsub.{u, v} o fun _ _ => a) = succ a :=
  bsup_const.{u, v} ho (succ a)


@[simp]
theorem blsub_one (f : ∀ a < (1 : Ordinal), Ordinal) : blsub 1 f = succ (f 0 zero_lt_one) :=
  bsup_one _


@[simp]
theorem blsub_id : ∀ o, (blsub.{u, u} o fun x _ => x) = o :=
  lsub_typein


theorem bsup_id_limit {o : Ordinal} : (∀ a < o, succ a < o) → (bsup.{u, u} o fun x _ => x) = o :=
  sup_typein_limit


@[simp]
theorem bsup_id_succ (o) : (bsup.{u, u} (succ o) fun x _ => x) = o :=
  sup_typein_succ


theorem blsub_le_of_brange_subset {o o'} {f : ∀ a < o, Ordinal} {g : ∀ a < o', Ordinal}
    (h : brange o f ⊆ brange o' g) : blsub.{u, max v w} o f ≤ blsub.{v, max u w} o' g :=
  bsup_le_of_brange_subset.{u, v, w} fun a ⟨b, hb, hb'⟩ => by
    /-
      o : Ordinal.{u}
      o' : Ordinal.{v}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max (max u v) w}
      g : (a : Ordinal.{v}) → LT.lt a o' → Ordinal.{max (max u v) w}
      h : HasSubset.Subset (o.brange f) (o'.brange g)
      a : Ordinal.{max (max u v) w}
      x✝ : Membership.mem (o.brange fun a ha => Order.succ (f a ha)) a
      b : Ordinal.{u}
      hb : LT.lt b o
      hb' : Eq ((fun a ha => Order.succ (f a ha)) b hb) a
      ⊢ Membership.mem (o'.brange fun a ha => Order.succ (g a ha)) a
    -/
    obtain ⟨c, hc, hc'⟩ := h ⟨b, hb, rfl⟩
    /-
      case intro.intro
      o : Ordinal.{u}
      o' : Ordinal.{v}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max (max u v) w}
      g : (a : Ordinal.{v}) → LT.lt a o' → Ordinal.{max (max u v) w}
      h : HasSubset.Subset (o.brange f) (o'.brange g)
      a : Ordinal.{max (max u v) w}
      x✝ : Membership.mem (o.brange fun a ha => Order.succ (f a ha)) a
      b : Ordinal.{u}
      hb : LT.lt b o
      hb' : Eq ((fun a ha => Order.succ (f a ha)) b hb) a
      c : Ordinal.{v}
      hc : LT.lt c o'
      hc' : Eq (g c hc) (f b hb)
      ⊢ Membership.mem (o'.brange fun a ha => Order.succ (g a ha)) a
    -/
    simp_rw [← hc'] at hb'
    /-
      case intro.intro
      o : Ordinal.{u}
      o' : Ordinal.{v}
      f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max (max u v) w}
      g : (a : Ordinal.{v}) → LT.lt a o' → Ordinal.{max (max u v) w}
      h : HasSubset.Subset (o.brange f) (o'.brange g)
      a : Ordinal.{max (max u v) w}
      x✝ : Membership.mem (o.brange fun a ha => Order.succ (f a ha)) a
      b : Ordinal.{u}
      hb : LT.lt b o
      c : Ordinal.{v}
      hc : LT.lt c o'
      hc' : Eq (g c hc) (f b hb)
      hb' : Eq (Order.succ (g c hc)) a
      ⊢ Membership.mem (o'.brange fun a ha => Order.succ (g a ha)) a
    -/
    exact ⟨c, hc, hb'⟩
    /-
      🎉 no goals
    -/


theorem blsub_eq_of_brange_eq {o o'} {f : ∀ a < o, Ordinal} {g : ∀ a < o', Ordinal}
    (h : { o | ∃ i hi, f i hi = o } = { o | ∃ i hi, g i hi = o }) :
    blsub.{u, max v w} o f = blsub.{v, max u w} o' g :=
  (blsub_le_of_brange_subset.{u, v, w} h.le).antisymm (blsub_le_of_brange_subset.{v, u, w} h.ge)


theorem bsup_comp {o o' : Ordinal.{max u v}} {f : ∀ a < o, Ordinal.{max u v w}}
    (hf : ∀ {i j} (hi) (hj), i ≤ j → f i hi ≤ f j hj) {g : ∀ a < o', Ordinal.{max u v}}
    (hg : blsub.{_, u} o' g = o) :
                                               /-
                                                 α : Type u_1
                                                 β : Type u_2
                                                 γ : Type u_3
                                                 r : α → α → Prop
                                                 s : β → β → Prop
                                                 t : γ → γ → Prop
                                                 o o' : Ordinal.{max u v}
                                                 f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
                                                 hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
                                                 g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
                                                 hg : Eq (o'.blsub g) o
                                                 a : Ordinal.{max u v}
                                                 ha : LT.lt a o'
                                                 ⊢ LT.lt (g a ha) o
                                               -/
    (bsup.{_, w} o' fun a ha => f (g a ha) (by rw [← hg]; apply lt_blsub)) = bsup.{_, w} o f := by
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    o o' : Ordinal.{max u v}
    f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
    hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
    g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
    hg : Eq (o'.blsub g) o
    ⊢ Eq (o'.bsup fun a ha => f (g a ha) ⋯) (o.bsup f)
  -/
  apply le_antisymm <;> refine bsup_le fun i hi => ?_
    /-
      case a
      o o' : Ordinal.{max u v}
      f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
      hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
      g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
      hg : Eq (o'.blsub g) o
      i : Ordinal.{max u v}
      hi : LT.lt i o'
      ⊢ LE.le (f (g i hi) ⋯) (o.bsup f)
    -/
  · apply le_bsup
    /-
      🎉 no goals
    -/
    /-
      case a
      o o' : Ordinal.{max u v}
      f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
      hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
      g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
      hg : Eq (o'.blsub g) o
      i : Ordinal.{max u v}
      hi : LT.lt i o
      ⊢ LE.le (f i hi) (o'.bsup fun a ha => f (g a ha) ⋯)
    -/
  · rw [← hg, lt_blsub_iff] at hi
    /-
      case a
      o o' : Ordinal.{max u v}
      f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
      hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
      g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
      hg : Eq (o'.blsub g) o
      i : Ordinal.{max u v}
      hi✝ : LT.lt i o
      hi : Exists fun i_1 => Exists fun hi => LE.le i (g i_1 hi)
      ⊢ LE.le (f i hi✝) (o'.bsup fun a ha => f (g a ha) ⋯)
    -/
    rcases hi with ⟨j, hj, hj'⟩
    /-
      case a.intro.intro
      o o' : Ordinal.{max u v}
      f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
      hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
      g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
      hg : Eq (o'.blsub g) o
      i : Ordinal.{max u v}
      hi : LT.lt i o
      j : Ordinal.{max u v}
      hj : LT.lt j o'
      hj' : LE.le i (g j hj)
      ⊢ LE.le (f i hi) (o'.bsup fun a ha => f (g a ha) ⋯)
    -/
    exact (hf _ _ hj').trans (le_bsup _ _ _)
    /-
      🎉 no goals
    -/


theorem blsub_comp {o o' : Ordinal.{max u v}} {f : ∀ a < o, Ordinal.{max u v w}}
    (hf : ∀ {i j} (hi) (hj), i ≤ j → f i hi ≤ f j hj) {g : ∀ a < o', Ordinal.{max u v}}
    (hg : blsub.{_, u} o' g = o) :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  γ : Type u_3
                                                  r : α → α → Prop
                                                  s : β → β → Prop
                                                  t : γ → γ → Prop
                                                  o o' : Ordinal.{max u v}
                                                  f : (a : Ordinal.{max u v}) → LT.lt a o → Ordinal.{max u v w}
                                                  hf : ∀ {i j : Ordinal.{max u v}} (hi : LT.lt i o) (hj : LT.lt j o), LE.le i j  …
                                                  g : (a : Ordinal.{max u v}) → LT.lt a o' → Ordinal.{max u v}
                                                  hg : Eq (o'.blsub g) o
                                                  a : Ordinal.{max u v}
                                                  ha : LT.lt a o'
                                                  ⊢ LT.lt (g a ha) o
                                                -/
    (blsub.{_, w} o' fun a ha => f (g a ha) (by rw [← hg]; apply lt_blsub)) = blsub.{_, w} o f :=
                                                           /-
                                                             🎉 no goals
                                                           -/
  @bsup_comp.{u, v, w} o _ (fun a ha => succ (f a ha))
    (fun {_ _} _ _ h => succ_le_succ_iff.2 (hf _ _ h)) g hg


theorem IsNormal.bsup_eq {f : Ordinal.{u} → Ordinal.{max u v}} (H : IsNormal f) {o : Ordinal.{u}}
    (h : IsLimit o) : (Ordinal.bsup.{_, v} o fun x _ => f x) = f o := by
  /-
    f : Ordinal.{u} → Ordinal.{max u v}
    H : Ordinal.IsNormal f
    o : Ordinal.{u}
    h : o.IsLimit
    ⊢ Eq (o.bsup fun x x_1 => f x) (f o)
  -/
  rw [← IsNormal.bsup.{u, u, v} H (fun x _ => x) h.ne_bot, bsup_id_limit fun _ ↦ h.succ_lt]
  /-
    🎉 no goals
  -/


theorem IsNormal.blsub_eq {f : Ordinal.{u} → Ordinal.{max u v}} (H : IsNormal f) {o : Ordinal.{u}}
    (h : IsLimit o) : (blsub.{_, v} o fun x _ => f x) = f o := by
  /-
    f : Ordinal.{u} → Ordinal.{max u v}
    H : Ordinal.IsNormal f
    o : Ordinal.{u}
    h : o.IsLimit
    ⊢ Eq (o.blsub fun x x_1 => f x) (f o)
  -/
  rw [← IsNormal.bsup_eq.{u, v} H h, bsup_eq_blsub_of_lt_succ_limit h]
  /-
    f : Ordinal.{u} → Ordinal.{max u v}
    H : Ordinal.IsNormal f
    o : Ordinal.{u}
    h : o.IsLimit
    ⊢ ∀ (a : Ordinal.{u}), LT.lt a o → LT.lt (f a) (f (Order.succ a))
  -/
  exact fun a _ => H.1 a
  /-
    🎉 no goals
  -/


theorem isNormal_iff_lt_succ_and_bsup_eq {f : Ordinal.{u} → Ordinal.{max u v}} :
    IsNormal f ↔ (∀ a, f a < f (succ a)) ∧ ∀ o, IsLimit o → (bsup.{_, v} o fun x _ => f x) = f o :=
  ⟨fun h => ⟨h.1, @IsNormal.bsup_eq f h⟩, fun ⟨h₁, h₂⟩ =>
    ⟨h₁, fun o ho a => by
      /-
        f : Ordinal.{u} → Ordinal.{max u v}
        x✝ : And (∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))) (∀ (o : Ordinal …
        h₁ : ∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))
        h₂ : ∀ (o : Ordinal.{u}), o.IsLimit → Eq (o.bsup fun x x_1 => f x) (f o)
        o : Ordinal.{u}
        ho : o.IsLimit
        a : Ordinal.{max u v}
        ⊢ Iff (LE.le (f o) a) (∀ (b : Ordinal.{u}), LT.lt b o → LE.le (f b) a)
      -/
      rw [← h₂ o ho]
      /-
        f : Ordinal.{u} → Ordinal.{max u v}
        x✝ : And (∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))) (∀ (o : Ordinal …
        h₁ : ∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))
        h₂ : ∀ (o : Ordinal.{u}), o.IsLimit → Eq (o.bsup fun x x_1 => f x) (f o)
        o : Ordinal.{u}
        ho : o.IsLimit
        a : Ordinal.{max u v}
        ⊢ Iff (LE.le (o.bsup fun x x_1 => f x) a) (∀ (b : Ordinal.{u}), LT.lt b o → LE …
      -/
      exact bsup_le_iff⟩⟩
      /-
        🎉 no goals
      -/


theorem isNormal_iff_lt_succ_and_blsub_eq {f : Ordinal.{u} → Ordinal.{max u v}} :
    IsNormal f ↔ (∀ a, f a < f (succ a)) ∧
      ∀ o, IsLimit o → (blsub.{_, v} o fun x _ => f x) = f o := by
  /-
    f : Ordinal.{u} → Ordinal.{max u v}
    ⊢ Iff (Ordinal.IsNormal f) (And (∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.su …
  -/
  rw [isNormal_iff_lt_succ_and_bsup_eq.{u, v}, and_congr_right_iff]
  /-
    f : Ordinal.{u} → Ordinal.{max u v}
    ⊢ (∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))) → Iff (∀ (o : Ordinal. …
  -/
  intro h
  /-
    f : Ordinal.{u} → Ordinal.{max u v}
    h : ∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))
    ⊢ Iff (∀ (o : Ordinal.{u}), o.IsLimit → Eq (o.bsup fun x x_1 => f x) (f o)) (∀ …
  -/
  constructor <;> intro H o ho <;> have := H o ho <;>
    /-
      case mp
      f : Ordinal.{u} → Ordinal.{max u v}
      h : ∀ (a : Ordinal.{u}), LT.lt (f a) (f (Order.succ a))
      H : ∀ (o : Ordinal.{u}), o.IsLimit → Eq (o.bsup fun x x_1 => f x) (f o)
      o : Ordinal.{u}
      ho : o.IsLimit
      this : Eq (o.bsup fun x x_1 => f x) (f o)
      ⊢ Eq (o.blsub fun x x_1 => f x) (f o)
    -/
    /-
      🎉 no goals
    -/
    rwa [← bsup_eq_blsub_of_lt_succ_limit ho fun a _ => h a] at *
    /-
      🎉 no goals
    -/


theorem IsNormal.eq_iff_zero_and_succ {f g : Ordinal.{u} → Ordinal.{u}} (hf : IsNormal f)
    (hg : IsNormal g) : f = g ↔ f 0 = g 0 ∧ ∀ a, f a = g a → f (succ a) = g (succ a) :=
               /-
                 f g : Ordinal.{u} → Ordinal.{u}
                 hf : Ordinal.IsNormal f
                 hg : Ordinal.IsNormal g
                 h : Eq f g
                 ⊢ And (Eq (f 0) (g 0)) (∀ (a : Ordinal.{u}), Eq (f a) (g a) → Eq (f (Order.suc …
               -/
  ⟨fun h => by simp [h], fun ⟨h₁, h₂⟩ =>
               /-
                 🎉 no goals
               -/
    funext fun a => by
      induction a using limitRecOn with
      | H₁ => solve_by_elim
      | H₂ => solve_by_elim
      | H₃ _ ho H =>
        rw [← IsNormal.bsup_eq.{u, u} hf ho, ← IsNormal.bsup_eq.{u, u} hg ho]
        congr
        ext b hb
        exact H b hb⟩


/-- A two-argument version of `Ordinal.blsub`.

Deprecated. If you need this value explicitly, write it in terms of `iSup`. If you just want an
upper bound for the image of `op`, use that `Iio a ×ˢ Iio b` is a small set. -/
@[deprecated "No deprecation message was provided."  (since := "2024-10-11")]
def blsub₂ (o₁ o₂ : Ordinal) (op : {a : Ordinal} → (a < o₁) → {b : Ordinal} → (b < o₂) → Ordinal) :
    Ordinal :=
  lsub (fun x : o₁.toType × o₂.toType => op (typein_lt_self x.1) (typein_lt_self x.2))


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-10-11")]
theorem lt_blsub₂ {o₁ o₂ : Ordinal}
    (op : {a : Ordinal} → (a < o₁) → {b : Ordinal} → (b < o₂) → Ordinal) {a b : Ordinal}
    (ha : a < o₁) (hb : b < o₂) : op ha hb < blsub₂ o₁ o₂ op := by
  convert lt_lsub _ (Prod.mk (enum (· < ·) ⟨a, by rwa [type_lt]⟩)
    (enum (· < ·) ⟨b, by rwa [type_lt]⟩))
  /-
    case h.e'_3
    o₁ : Ordinal.{u_4}
    o₂ : Ordinal.{u_5}
    op : {a : Ordinal.{u_4}} → LT.lt a o₁ → {b : Ordinal.{u_5}} → LT.lt b o₂ → Ord …
    a : Ordinal.{u_4}
    b : Ordinal.{u_5}
    ha : LT.lt a o₁
    hb : LT.lt b o₂
    ⊢ Eq (op ha hb) ((fun {a} => op) ⋯ ⋯)
  -/
  simp only [typein_enum]
  /-
    🎉 no goals
  -/


/-- The minimum excluded ordinal in a family of ordinals. -/
@[deprecated "use sInf sᶜ instead" (since := "2024-09-20")]
def mex {ι : Type u} (f : ι → Ordinal.{max u v}) : Ordinal :=
  sInf (Set.range f)ᶜ


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem mex_not_mem_range {ι : Type u} (f : ι → Ordinal.{max u v}) : mex.{_, v} f ∉ Set.range f :=
  csInf_mem (nonempty_compl_range.{_, v} f)


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem le_mex_of_forall {ι : Type u} {f : ι → Ordinal.{max u v}} {a : Ordinal}
    (H : ∀ b < a, ∃ i, f i = b) : a ≤ mex.{_, v} f := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    a : Ordinal.{max u v}
    H : ∀ (b : Ordinal.{max u v}), LT.lt b a → Exists fun i => Eq (f i) b
    ⊢ LE.le a (Ordinal.mex f)
  -/
  by_contra! h
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    a : Ordinal.{max u v}
    H : ∀ (b : Ordinal.{max u v}), LT.lt b a → Exists fun i => Eq (f i) b
    h : LT.lt (Ordinal.mex f) a
    ⊢ False
  -/
  exact mex_not_mem_range f (H _ h)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem ne_mex {ι : Type u} (f : ι → Ordinal.{max u v}) : ∀ i, f i ≠ mex.{_, v} f := by
  /-
    ι : Type u
    f : ι → Ordinal.{max u v}
    ⊢ ∀ (i : ι), Ne (f i) (Ordinal.mex f)
  -/
  simpa using mex_not_mem_range.{_, v} f
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem mex_le_of_ne {ι} {f : ι → Ordinal} {a} (ha : ∀ i, f i ≠ a) : mex f ≤ a :=
                /-
                  ι : Type u_4
                  f : ι → Ordinal.{max u_5 u_4}
                  a : Ordinal.{max u_5 u_4}
                  ha : ∀ (i : ι), Ne (f i) a
                  ⊢ Membership.mem (HasCompl.compl (Set.range f)) a
                -/
  csInf_le' (by simp [ha])
                /-
                  🎉 no goals
                -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem exists_of_lt_mex {ι} {f : ι → Ordinal} {a} (ha : a < mex f) : ∃ i, f i = a := by
  /-
    ι : Type u_4
    f : ι → Ordinal.{max u_5 u_4}
    a : Ordinal.{max u_4 u_5}
    ha : LT.lt a (Ordinal.mex f)
    ⊢ Exists fun i => Eq (f i) a
  -/
  by_contra! ha'
  /-
    ι : Type u_4
    f : ι → Ordinal.{max u_5 u_4}
    a : Ordinal.{max u_4 u_5}
    ha : LT.lt a (Ordinal.mex f)
    ha' : ∀ (i : ι), Ne (f i) a
    ⊢ False
  -/
  exact ha.not_le (mex_le_of_ne ha')
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem mex_le_lsub {ι : Type u} (f : ι → Ordinal.{max u v}) : mex.{_, v} f ≤ lsub.{_, v} f :=
  csInf_le' (lsub_not_mem_range f)


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem mex_monotone {α β : Type u} {f : α → Ordinal.{max u v}} {g : β → Ordinal.{max u v}}
    (h : Set.range f ⊆ Set.range g) : mex.{_, v} f ≤ mex.{_, v} g := by
  /-
    α β : Type u
    f : α → Ordinal.{max u v}
    g : β → Ordinal.{max u v}
    h : HasSubset.Subset (Set.range f) (Set.range g)
    ⊢ LE.le (Ordinal.mex f) (Ordinal.mex g)
  -/
  refine mex_le_of_ne fun i hi => ?_
  /-
    α β : Type u
    f : α → Ordinal.{max u v}
    g : β → Ordinal.{max u v}
    h : HasSubset.Subset (Set.range f) (Set.range g)
    i : α
    hi : Eq (f i) (Ordinal.mex g)
    ⊢ False
  -/
  cases' h ⟨i, rfl⟩ with j hj
  /-
    case intro
    α β : Type u
    f : α → Ordinal.{max u v}
    g : β → Ordinal.{max u v}
    h : HasSubset.Subset (Set.range f) (Set.range g)
    i : α
    hi : Eq (f i) (Ordinal.mex g)
    j : β
    hj : Eq (g j) (f i)
    ⊢ False
  -/
  rw [← hj] at hi
  /-
    case intro
    α β : Type u
    f : α → Ordinal.{max u v}
    g : β → Ordinal.{max u v}
    h : HasSubset.Subset (Set.range f) (Set.range g)
    i : α
    j : β
    hi : Eq (g j) (Ordinal.mex g)
    hj : Eq (g j) (f i)
    ⊢ False
  -/
  exact ne_mex g j hi
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated sInf_compl_lt_ord_succ (since := "2024-09-20")]
theorem mex_lt_ord_succ_mk {ι : Type u} (f : ι → Ordinal.{u}) :
    mex.{_, u} f < (succ #ι).ord := by
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    ⊢ LT.lt (Ordinal.mex f) (Order.succ (Cardinal.mk ι)).ord
  -/
  by_contra! h
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    h : LE.le (Order.succ (Cardinal.mk ι)).ord (Ordinal.mex f)
    ⊢ False
  -/
  apply (lt_succ #ι).not_le
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    h : LE.le (Order.succ (Cardinal.mk ι)).ord (Ordinal.mex f)
    ⊢ LE.le (Order.succ (Cardinal.mk ι)) (Cardinal.mk ι)
  -/
  have H := fun a => exists_of_lt_mex ((typein_lt_self a).trans_le h)
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    h : LE.le (Order.succ (Cardinal.mk ι)).ord (Ordinal.mex f)
    H : ∀ (a : (Order.succ (Cardinal.mk ι)).ord.toType), Exists fun i => Eq (f i)  …
    ⊢ LE.le (Order.succ (Cardinal.mk ι)) (Cardinal.mk ι)
  -/
  let g : (succ #ι).ord.toType → ι := fun a => Classical.choose (H a)
  have hg : Injective g := fun a b h' => by
    have Hf : ∀ x, f (g x) =
        typein ((· < ·) : (succ #ι).ord.toType → (succ #ι).ord.toType → Prop) x :=
      fun a => Classical.choose_spec (H a)
    apply_fun f at h'
    rwa [Hf, Hf, typein_inj] at h'
  /-
    ι : Type u
    f : ι → Ordinal.{u}
    h : LE.le (Order.succ (Cardinal.mk ι)).ord (Ordinal.mex f)
    H : ∀ (a : (Order.succ (Cardinal.mk ι)).ord.toType), Exists fun i => Eq (f i)  …
    g : (Order.succ (Cardinal.mk ι)).ord.toType → ι := fun a => Classical.choose ⋯
    hg : Function.Injective g
    ⊢ LE.le (Order.succ (Cardinal.mk ι)) (Cardinal.mk ι)
  -/
  convert Cardinal.mk_le_of_injective hg
  /-
    case h.e'_3
    ι : Type u
    f : ι → Ordinal.{u}
    h : LE.le (Order.succ (Cardinal.mk ι)).ord (Ordinal.mex f)
    H : ∀ (a : (Order.succ (Cardinal.mk ι)).ord.toType), Exists fun i => Eq (f i)  …
    g : (Order.succ (Cardinal.mk ι)).ord.toType → ι := fun a => Classical.choose ⋯
    hg : Function.Injective g
    ⊢ Eq (Order.succ (Cardinal.mk ι)) (Cardinal.mk (Order.succ (Cardinal.mk ι)).or …
  -/
  rw [Cardinal.mk_ord_toType (succ #ι)]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
/-- The minimum excluded ordinal of a family of ordinals indexed by the set of ordinals less than
    some `o : Ordinal.{u}`. This is a special case of `mex` over the family provided by
    `familyOfBFamily`.

    This is to `mex` as `bsup` is to `sup`. -/
@[deprecated "use sInf sᶜ instead" (since := "2024-09-20")]
def bmex (o : Ordinal) (f : ∀ a < o, Ordinal) : Ordinal :=
  mex (familyOfBFamily o f)


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem bmex_not_mem_brange {o : Ordinal} (f : ∀ a < o, Ordinal) : bmex o f ∉ brange o f := by
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_4 u_5}
    ⊢ Not (Membership.mem (o.brange f) (o.bmex f))
  -/
  rw [← range_familyOfBFamily]
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_4 u_5}
    ⊢ Not (Membership.mem (Set.range (o.familyOfBFamily f)) (o.bmex f))
  -/
  apply mex_not_mem_range
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem le_bmex_of_forall {o : Ordinal} (f : ∀ a < o, Ordinal) {a : Ordinal}
    (H : ∀ b < a, ∃ i hi, f i hi = b) : a ≤ bmex o f := by
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_4 u_5}
    a : Ordinal.{max u_4 u_5}
    H : ∀ (b : Ordinal.{max u_4 u_5}), LT.lt b a → Exists fun i => Exists fun hi = …
    ⊢ LE.le a (o.bmex f)
  -/
  by_contra! h
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_4 u_5}
    a : Ordinal.{max u_4 u_5}
    H : ∀ (b : Ordinal.{max u_4 u_5}), LT.lt b a → Exists fun i => Exists fun hi = …
    h : LT.lt (o.bmex f) a
    ⊢ False
  -/
  exact bmex_not_mem_brange f (H _ h)
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem ne_bmex {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) {i} (hi) :
    f i hi ≠ bmex.{_, v} o f := by
  convert (config := {transparency := .default})
    ne_mex.{_, v} (familyOfBFamily o f) (enum (α := o.toType) (· < ·) ⟨i, by rwa [type_lt]⟩) using 2
  -- Porting note: `familyOfBFamily_enum` → `typein_enum`
  /-
    case h.e'_2.h.e'_1
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
    i : Ordinal.{u}
    hi : LT.lt i o
    ⊢ Eq i ((Ordinal.typein fun x1 x2 => LT.lt x1 x2).toRelEmbedding ((Ordinal.enu …
  -/
  rw [typein_enum]
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem bmex_le_of_ne {o : Ordinal} {f : ∀ a < o, Ordinal} {a} (ha : ∀ i hi, f i hi ≠ a) :
    bmex o f ≤ a :=
  mex_le_of_ne fun _i => ha _ _


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem exists_of_lt_bmex {o : Ordinal} {f : ∀ a < o, Ordinal} {a} (ha : a < bmex o f) :
    ∃ i hi, f i hi = a := by
  /-
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_4 u_5}
    a : Ordinal.{max u_5 u_4}
    ha : LT.lt a (o.bmex f)
    ⊢ Exists fun i => Exists fun hi => Eq (f i hi) a
  -/
  cases' exists_of_lt_mex ha with i hi
  /-
    case intro
    o : Ordinal.{u_4}
    f : (a : Ordinal.{u_4}) → LT.lt a o → Ordinal.{max u_4 u_5}
    a : Ordinal.{max u_5 u_4}
    ha : LT.lt a (o.bmex f)
    i : o.toType
    hi : Eq (o.familyOfBFamily f i) a
    ⊢ Exists fun i => Exists fun hi => Eq (f i hi) a
  -/
  exact ⟨_, typein_lt_self i, hi⟩
  /-
    🎉 no goals
  -/


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem bmex_le_blsub {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{max u v}) :
    bmex.{_, v} o f ≤ blsub.{_, v} o f :=
  mex_le_lsub _


set_option linter.deprecated false in
@[deprecated "No deprecation message was provided."  (since := "2024-09-20")]
theorem bmex_monotone {o o' : Ordinal.{u}}
    {f : ∀ a < o, Ordinal.{max u v}} {g : ∀ a < o', Ordinal.{max u v}}
    (h : brange o f ⊆ brange o' g) : bmex.{_, v} o f ≤ bmex.{_, v} o' g :=
                   /-
                     o o' : Ordinal.{u}
                     f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{max u v}
                     g : (a : Ordinal.{u}) → LT.lt a o' → Ordinal.{max u v}
                     h : HasSubset.Subset (o.brange f) (o'.brange g)
                     ⊢ HasSubset.Subset (Set.range (o.familyOfBFamily f)) (Set.range (o'.familyOfBF …
                   -/
  mex_monotone (by rwa [range_familyOfBFamily, range_familyOfBFamily])
                   /-
                     🎉 no goals
                   -/


set_option linter.deprecated false in
@[deprecated sInf_compl_lt_ord_succ (since := "2024-09-20")]
theorem bmex_lt_ord_succ_card {o : Ordinal.{u}} (f : ∀ a < o, Ordinal.{u}) :
    bmex.{_, u} o f < (succ o.card).ord := by
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
    ⊢ LT.lt (o.bmex f) (Order.succ o.card).ord
  -/
  rw [← mk_toType]
  /-
    o : Ordinal.{u}
    f : (a : Ordinal.{u}) → LT.lt a o → Ordinal.{u}
    ⊢ LT.lt (o.bmex f) (Order.succ (Cardinal.mk o.toType)).ord
  -/
  exact mex_lt_ord_succ_mk (familyOfBFamily o f)
  /-
    🎉 no goals
  -/


theorem not_surjective_of_ordinal {α : Type u} (f : α → Ordinal.{u}) : ¬Surjective f := fun h =>
  Ordinal.lsub_not_mem_range.{u, u} f (h _)


theorem not_injective_of_ordinal {α : Type u} (f : Ordinal.{u} → α) : ¬Injective f := fun h =>
  not_surjective_of_ordinal _ (invFun_surjective h)


theorem not_surjective_of_ordinal_of_small {α : Type v} [Small.{u} α] (f : α → Ordinal.{u}) :
    ¬Surjective f := fun h => not_surjective_of_ordinal _ (h.comp (equivShrink _).symm.surjective)


theorem not_injective_of_ordinal_of_small {α : Type v} [Small.{u} α] (f : Ordinal.{u} → α) :
    ¬Injective f := fun h => not_injective_of_ordinal _ ((equivShrink _).injective.comp h)


/-- The type of ordinals in universe `u` is not `Small.{u}`. This is the type-theoretic analog of
the Burali-Forti paradox. -/
theorem not_small_ordinal : ¬Small.{u} Ordinal.{max u v} := fun h =>
  @not_injective_of_ordinal_of_small _ h _ fun _a _b => Ordinal.lift_inj.{v, u}.1


theorem Ordinal.not_bddAbove_compl_of_small (s : Set Ordinal.{u}) [hs : Small.{u} s] :
    ¬BddAbove sᶜ := by
  /-
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    ⊢ Not (BddAbove (HasCompl.compl s))
  -/
  rw [bddAbove_iff_small]
  /-
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    ⊢ Not (Small.{u, u + 1} ↑(HasCompl.compl s))
  -/
  intro h
  /-
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    h : Small.{u, u + 1} ↑(HasCompl.compl s)
    ⊢ False
  -/
  have := small_union s sᶜ
  /-
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    h : Small.{u, u + 1} ↑(HasCompl.compl s)
    this : Small.{u, u + 1} ↑(Union.union s (HasCompl.compl s))
    ⊢ False
  -/
  rw [union_compl_self, small_univ_iff] at this
  /-
    s : Set Ordinal.{u}
    hs : Small.{u, u + 1} ↑s
    h : Small.{u, u + 1} ↑(HasCompl.compl s)
    this : Small.{u, u + 1} Ordinal.{u}
    ⊢ False
  -/
  exact not_small_ordinal this
  /-
    🎉 no goals
  -/


instance instCharZero : CharZero Ordinal := by
  /-
    ⊢ CharZero Ordinal.{u_1}
  -/
  refine ⟨fun a b h ↦ ?_⟩
  /-
    a b : Nat
    h : Eq ↑a ↑b
    ⊢ Eq a b
  -/
  rwa [← Cardinal.ord_nat, ← Cardinal.ord_nat, Cardinal.ord_inj, Nat.cast_inj] at h
  /-
    🎉 no goals
  -/


@[simp]
theorem one_add_natCast (m : ℕ) : 1 + (m : Ordinal) = succ m := by
  /-
    m : Nat
    ⊢ Eq (HAdd.hAdd 1 ↑m) ↑(Order.succ m)
  -/
  rw [← Nat.cast_one, ← Nat.cast_add, add_comm]
  /-
    m : Nat
    ⊢ Eq ↑(HAdd.hAdd m 1) ↑(Order.succ m)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias one_add_nat_cast := one_add_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem one_add_ofNat (m : ℕ) [m.AtLeastTwo] :
    1 + (no_index (OfNat.ofNat m : Ordinal)) = Order.succ (OfNat.ofNat m : Ordinal) :=
  one_add_natCast m


@[simp, norm_cast]
theorem natCast_mul (m : ℕ) : ∀ n : ℕ, ((m * n : ℕ) : Ordinal) = m * n
            /-
              m : Nat
              ⊢ Eq (↑(HMul.hMul m 0)) (HMul.hMul ↑m ↑0)
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  m n : Nat
                  ⊢ Eq (↑(HMul.hMul m (HAdd.hAdd n 1))) (HMul.hMul ↑m ↑(HAdd.hAdd n 1))
                -/
  | n + 1 => by rw [Nat.mul_succ, Nat.cast_add, natCast_mul m n, Nat.cast_succ, mul_add_one]
                /-
                  🎉 no goals
                -/


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_mul := natCast_mul


@[deprecated Nat.cast_le (since := "2024-10-17")]
theorem natCast_le {m n : ℕ} : (m : Ordinal) ≤ n ↔ m ≤ n := Nat.cast_le


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_le := natCast_le


@[deprecated Nat.cast_inj (since := "2024-10-17")]
theorem natCast_inj {m n : ℕ} : (m : Ordinal) = n ↔ m = n := Nat.cast_inj


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_inj := natCast_inj


@[deprecated Nat.cast_lt (since := "2024-10-17")]
theorem natCast_lt {m n : ℕ} : (m : Ordinal) < n ↔ m < n := Nat.cast_lt


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_lt := natCast_lt


@[deprecated Nat.cast_eq_zero (since := "2024-10-17")]
theorem natCast_eq_zero {n : ℕ} : (n : Ordinal) = 0 ↔ n = 0 := Nat.cast_eq_zero


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_eq_zero := natCast_eq_zero


@[deprecated Nat.cast_ne_zero (since := "2024-10-17")]
theorem natCast_ne_zero {n : ℕ} : (n : Ordinal) ≠ 0 ↔ n ≠ 0 := Nat.cast_ne_zero


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_ne_zero := natCast_ne_zero


@[deprecated Nat.cast_pos' (since := "2024-10-17")]
theorem natCast_pos {n : ℕ} : (0 : Ordinal) < n ↔ 0 < n := Nat.cast_pos'


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_pos := natCast_pos


@[simp, norm_cast]
theorem natCast_sub (m n : ℕ) : ((m - n : ℕ) : Ordinal) = m - n := by
  /-
    m n : Nat
    ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
  -/
  rcases le_total m n with h | h
    /-
      case inl
      m n : Nat
      h : LE.le m n
      ⊢ Eq (↑(HSub.hSub m n)) (HSub.hSub ↑m ↑n)
    -/
  · rw [tsub_eq_zero_iff_le.2 h, Ordinal.sub_eq_zero_iff_le.2 (Nat.cast_le.2 h), Nat.cast_zero]
    /-
      🎉 no goals
    -/
  · rw [← add_left_cancel_iff (a := ↑n), ← Nat.cast_add, add_tsub_cancel_of_le h,
      Ordinal.add_sub_cancel_of_le (Nat.cast_le.2 h)]


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_sub := natCast_sub


@[simp, norm_cast]
theorem natCast_div (m n : ℕ) : ((m / n : ℕ) : Ordinal) = m / n := by
  /-
    m n : Nat
    ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case inl
      m : Nat
      ⊢ Eq (↑(HDiv.hDiv m 0)) (HDiv.hDiv ↑m ↑0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      m n : Nat
      hn : Ne n 0
      ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
    -/
  · have hn' : (n : Ordinal) ≠ 0 := Nat.cast_ne_zero.2 hn
    /-
      case inr
      m n : Nat
      hn : Ne n 0
      hn' : Ne (↑n) 0
      ⊢ Eq (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
    -/
    apply le_antisymm
      /-
        case inr.a
        m n : Nat
        hn : Ne n 0
        hn' : Ne (↑n) 0
        ⊢ LE.le (↑(HDiv.hDiv m n)) (HDiv.hDiv ↑m ↑n)
      -/
    · rw [le_div hn', ← natCast_mul, Nat.cast_le, mul_comm]
      /-
        case inr.a
        m n : Nat
        hn : Ne n 0
        hn' : Ne (↑n) 0
        ⊢ LE.le (HMul.hMul (HDiv.hDiv m n) n) m
      -/
      apply Nat.div_mul_le_self
      /-
        🎉 no goals
      -/
    · rw [div_le hn', ← add_one_eq_succ, ← Nat.cast_succ, ← natCast_mul, Nat.cast_lt, mul_comm,
        ← Nat.div_lt_iff_lt_mul (Nat.pos_of_ne_zero hn)]
      /-
        case inr.a
        m n : Nat
        hn : Ne n 0
        hn' : Ne (↑n) 0
        ⊢ LT.lt (HDiv.hDiv m n) (HDiv.hDiv m n).succ
      -/
      apply Nat.lt_succ_self
      /-
        🎉 no goals
      -/


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_div := natCast_div


@[simp, norm_cast]
theorem natCast_mod (m n : ℕ) : ((m % n : ℕ) : Ordinal) = m % n := by
  rw [← add_left_cancel_iff, div_add_mod, ← natCast_div, ← natCast_mul, ← Nat.cast_add,
    Nat.div_add_mod]


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias nat_cast_mod := natCast_mod


@[simp]
theorem lift_natCast : ∀ n : ℕ, lift.{u, v} n = n
            /-
              ⊢ Eq (Ordinal.lift.{u, v} ↑0) ↑0
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
                /-
                  n : Nat
                  ⊢ Eq (Ordinal.lift.{u, v} ↑(HAdd.hAdd n 1)) ↑(HAdd.hAdd n 1)
                -/
  | n + 1 => by simp [lift_natCast n]
                /-
                  🎉 no goals
                -/


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias lift_nat_cast := lift_natCast

-- See note [no_index around OfNat.ofNat]

@[simp]
theorem lift_ofNat (n : ℕ) [n.AtLeastTwo] :
    lift.{u, v} (no_index (OfNat.ofNat n)) = OfNat.ofNat n :=
  lift_natCast n


theorem lt_add_of_limit {a b c : Ordinal.{u}} (h : IsLimit c) :
    a < b + c ↔ ∃ c' < c, a < b + c' := by
  -- Porting note: `bex_def` is required.
  /-
    a b c : Ordinal.{u}
    h : c.IsLimit
    ⊢ Iff (LT.lt a (HAdd.hAdd b c)) (Exists fun c' => And (LT.lt c' c) (LT.lt a (H …
  -/
  rw [← IsNormal.bsup_eq.{u, u} (isNormal_add_right b) h, lt_bsup, bex_def]
  /-
    🎉 no goals
  -/


theorem lt_omega0 {o : Ordinal} : o < ω ↔ ∃ n : ℕ, o = n := by
  /-
    o : Ordinal.{u_1}
    ⊢ Iff (LT.lt o Ordinal.omega0) (Exists fun n => Eq o ↑n)
  -/
  simp_rw [← Cardinal.ord_aleph0, Cardinal.lt_ord, lt_aleph0, card_eq_nat]
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias lt_omega := lt_omega0


theorem nat_lt_omega0 (n : ℕ) : ↑n < ω :=
  lt_omega0.2 ⟨_, rfl⟩


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias nat_lt_omega := nat_lt_omega0


theorem eq_nat_or_omega0_le (o : Ordinal) : (∃ n : ℕ, o = n) ∨ ω ≤ o := by
  /-
    o : Ordinal.{u_1}
    ⊢ Or (Exists fun n => Eq o ↑n) (LE.le Ordinal.omega0 o)
  -/
  obtain ho | ho := lt_or_le o ω
    /-
      case inl
      o : Ordinal.{u_1}
      ho : LT.lt o Ordinal.omega0
      ⊢ Or (Exists fun n => Eq o ↑n) (LE.le Ordinal.omega0 o)
    -/
  · exact Or.inl <| lt_omega0.1 ho
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u_1}
      ho : LE.le Ordinal.omega0 o
      ⊢ Or (Exists fun n => Eq o ↑n) (LE.le Ordinal.omega0 o)
    -/
  · exact Or.inr ho
    /-
      🎉 no goals
    -/


theorem omega0_pos : 0 < ω :=
  nat_lt_omega0 0


theorem omega0_ne_zero : ω ≠ 0 :=
  omega0_pos.ne'


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias omega_ne_zero := omega0_ne_zero


                                    /-
                                      ⊢ LT.lt 1 Ordinal.omega0
                                    -/
theorem one_lt_omega0 : 1 < ω := by simpa only [Nat.cast_one] using nat_lt_omega0 1
                                    /-
                                      🎉 no goals
                                    -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias one_lt_omega := one_lt_omega0


theorem isLimit_omega0 : IsLimit ω := by
  /-
    ⊢ Ordinal.omega0.IsLimit
  -/
  rw [isLimit_iff, isSuccPrelimit_iff_succ_lt]
  /-
    ⊢ And (Ne Ordinal.omega0 0) (∀ (a : Ordinal.{u_1}), LT.lt a Ordinal.omega0 → L …
  -/
  refine ⟨omega0_ne_zero, fun o h => ?_⟩
  /-
    o : Ordinal.{u_1}
    h : LT.lt o Ordinal.omega0
    ⊢ LT.lt (Order.succ o) Ordinal.omega0
  -/
  obtain ⟨n, rfl⟩ := lt_omega0.1 h
  /-
    case intro
    n : Nat
    h : LT.lt (↑n) Ordinal.omega0
    ⊢ LT.lt (Order.succ ↑n) Ordinal.omega0
  -/
  exact nat_lt_omega0 (n + 1)
  /-
    🎉 no goals
  -/


@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
alias omega0_isLimit := isLimit_omega0


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias omega_isLimit := isLimit_omega0


theorem omega0_le {o : Ordinal} : ω ≤ o ↔ ∀ n : ℕ, ↑n ≤ o :=
  ⟨fun h n => (nat_lt_omega0 _).le.trans h, fun H =>
    le_of_forall_lt fun a h => by
      /-
        o : Ordinal.{u_1}
        H : ∀ (n : Nat), LE.le (↑n) o
        a : Ordinal.{u_1}
        h : LT.lt a Ordinal.omega0
        ⊢ LT.lt a o
      -/
      let ⟨n, e⟩ := lt_omega0.1 h
      /-
        o : Ordinal.{u_1}
        H : ∀ (n : Nat), LE.le (↑n) o
        a : Ordinal.{u_1}
        h : LT.lt a Ordinal.omega0
        n : Nat
        e : Eq a ↑n
        ⊢ LT.lt a o
      -/
      rw [e, ← succ_le_iff]; exact H (n + 1)⟩
                             /-
                               🎉 no goals
                             -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias omega_le := omega0_le


@[simp]
theorem iSup_natCast : iSup Nat.cast = ω :=
  (Ordinal.iSup_le fun n => (nat_lt_omega0 n).le).antisymm <| omega0_le.2 <| Ordinal.le_iSup _


set_option linter.deprecated false in
@[deprecated iSup_natCast (since := "2024-04-17")]
theorem sup_natCast : sup Nat.cast = ω :=
  iSup_natCast


@[deprecated "No deprecation message was provided."  (since := "2024-04-17")]
alias sup_nat_cast := sup_natCast


theorem nat_lt_limit {o} (h : IsLimit o) : ∀ n : ℕ, ↑n < o
  | 0 => h.pos
  | n + 1 => h.succ_lt (nat_lt_limit h n)


theorem omega0_le_of_isLimit {o} (h : IsLimit o) : ω ≤ o :=
  omega0_le.2 fun n => le_of_lt <| nat_lt_limit h n


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias omega_le_of_isLimit := omega0_le_of_isLimit


theorem natCast_add_omega0 (n : ℕ) : n + ω = ω := by
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd (↑n) Ordinal.omega0) Ordinal.omega0
  -/
  refine le_antisymm (le_of_forall_lt fun a ha ↦ ?_) (le_add_left _ _)
  /-
    n : Nat
    a : Ordinal.{u_1}
    ha : LT.lt a (HAdd.hAdd (↑n) Ordinal.omega0)
    ⊢ LT.lt a Ordinal.omega0
  -/
  obtain ⟨b, hb', hb⟩ := (lt_add_iff omega0_ne_zero).1 ha
  /-
    case intro.intro
    n : Nat
    a : Ordinal.{u_1}
    ha : LT.lt a (HAdd.hAdd (↑n) Ordinal.omega0)
    b : Ordinal.{u_1}
    hb' : LT.lt b Ordinal.omega0
    hb : LE.le a (HAdd.hAdd (↑n) b)
    ⊢ LT.lt a Ordinal.omega0
  -/
  obtain ⟨m, rfl⟩ := lt_omega0.1 hb'
  /-
    case intro.intro.intro
    n : Nat
    a : Ordinal.{u_1}
    ha : LT.lt a (HAdd.hAdd (↑n) Ordinal.omega0)
    m : Nat
    hb' : LT.lt (↑m) Ordinal.omega0
    hb : LE.le a (HAdd.hAdd ↑n ↑m)
    ⊢ LT.lt a Ordinal.omega0
  -/
  apply hb.trans_lt
  /-
    case intro.intro.intro
    n : Nat
    a : Ordinal.{u_1}
    ha : LT.lt a (HAdd.hAdd (↑n) Ordinal.omega0)
    m : Nat
    hb' : LT.lt (↑m) Ordinal.omega0
    hb : LE.le a (HAdd.hAdd ↑n ↑m)
    ⊢ LT.lt (HAdd.hAdd ↑n ↑m) Ordinal.omega0
  -/
  exact_mod_cast nat_lt_omega0 (n + m)
  /-
    🎉 no goals
  -/


theorem one_add_omega0 : 1 + ω = ω :=
  mod_cast natCast_add_omega0 1


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias one_add_omega := one_add_omega0


theorem add_omega0 {a : Ordinal} (h : a < ω) : a + ω = ω := by
  /-
    a : Ordinal.{u_1}
    h : LT.lt a Ordinal.omega0
    ⊢ Eq (HAdd.hAdd a Ordinal.omega0) Ordinal.omega0
  -/
  obtain ⟨n, rfl⟩ := lt_omega0.1 h
  /-
    case intro
    n : Nat
    h : LT.lt (↑n) Ordinal.omega0
    ⊢ Eq (HAdd.hAdd (↑n) Ordinal.omega0) Ordinal.omega0
  -/
  exact natCast_add_omega0 n
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-30")]
alias add_omega := add_omega0


@[simp]
theorem natCast_add_of_omega0_le {o} (h : ω ≤ o) (n : ℕ) : n + o = o := by
  /-
    o : Ordinal.{u_1}
    h : LE.le Ordinal.omega0 o
    n : Nat
    ⊢ Eq (HAdd.hAdd (↑n) o) o
  -/
  rw [← Ordinal.add_sub_cancel_of_le h, ← add_assoc, natCast_add_omega0]
  /-
    🎉 no goals
  -/


@[simp]
theorem one_add_of_omega0_le {o} (h : ω ≤ o) : 1 + o = o :=
  mod_cast natCast_add_of_omega0_le h 1


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias one_add_of_omega_le := one_add_of_omega0_le


theorem isLimit_iff_omega0_dvd {a : Ordinal} : IsLimit a ↔ a ≠ 0 ∧ ω ∣ a := by
  /-
    a : Ordinal.{u_1}
    ⊢ Iff a.IsLimit (And (Ne a 0) (Dvd.dvd Ordinal.omega0 a))
  -/
  refine ⟨fun l => ⟨l.ne_zero, ⟨a / ω, le_antisymm ?_ (mul_div_le _ _)⟩⟩, fun h => ?_⟩
    /-
      case refine_1
      a : Ordinal.{u_1}
      l : a.IsLimit
      ⊢ LE.le a (HMul.hMul Ordinal.omega0 (HDiv.hDiv a Ordinal.omega0))
    -/
  · refine (limit_le l).2 fun x hx => le_of_lt ?_
    rw [← div_lt omega0_ne_zero, ← succ_le_iff, le_div omega0_ne_zero, mul_succ,
      add_le_of_limit isLimit_omega0]
    /-
      case refine_1
      a : Ordinal.{u_1}
      l : a.IsLimit
      x : Ordinal.{u_1}
      hx : LT.lt x a
      ⊢ ∀ (b' : Ordinal.{u_1}), LT.lt b' Ordinal.omega0 → LE.le (HAdd.hAdd (HMul.hMu …
    -/
    intro b hb
    /-
      case refine_1
      a : Ordinal.{u_1}
      l : a.IsLimit
      x : Ordinal.{u_1}
      hx : LT.lt x a
      b : Ordinal.{u_1}
      hb : LT.lt b Ordinal.omega0
      ⊢ LE.le (HAdd.hAdd (HMul.hMul Ordinal.omega0 (HDiv.hDiv x Ordinal.omega0)) b) a
    -/
    rcases lt_omega0.1 hb with ⟨n, rfl⟩
    exact
      (add_le_add_right (mul_div_le _ _) _).trans
        (lt_sub.1 <| nat_lt_limit (isLimit_sub l hx) _).le
    /-
      case refine_2
      a : Ordinal.{u_1}
      h : And (Ne a 0) (Dvd.dvd Ordinal.omega0 a)
      ⊢ a.IsLimit
    -/
  · rcases h with ⟨a0, b, rfl⟩
    /-
      case refine_2.intro.intro
      b : Ordinal.{u_1}
      a0 : Ne (HMul.hMul Ordinal.omega0 b) 0
      ⊢ (HMul.hMul Ordinal.omega0 b).IsLimit
    -/
    refine isLimit_mul_left isLimit_omega0 (Ordinal.pos_iff_ne_zero.2 <| mt ?_ a0)
    /-
      case refine_2.intro.intro
      b : Ordinal.{u_1}
      a0 : Ne (HMul.hMul Ordinal.omega0 b) 0
      ⊢ Eq b 0 → Eq (HMul.hMul Ordinal.omega0 b) 0
    -/
    intro e
    /-
      case refine_2.intro.intro
      b : Ordinal.{u_1}
      a0 : Ne (HMul.hMul Ordinal.omega0 b) 0
      e : Eq b 0
      ⊢ Eq (HMul.hMul Ordinal.omega0 b) 0
    -/
    simp only [e, mul_zero]
    /-
      🎉 no goals
    -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias isLimit_iff_omega_dvd := isLimit_iff_omega0_dvd


theorem IsNormal.apply_omega0 {f : Ordinal.{u} → Ordinal.{v}} (hf : IsNormal f) :
                             /-
                               f : Ordinal.{u} → Ordinal.{v}
                               hf : Ordinal.IsNormal f
                               ⊢ Eq (iSup fun n => f ↑n) (f Ordinal.omega0)
                             -/
    ⨆ n : ℕ, f n = f ω := by rw [← iSup_natCast, hf.map_iSup]
                             /-
                               🎉 no goals
                             -/


@[deprecated "No deprecation message was provided."  (since := "2024-09-30")]
alias IsNormal.apply_omega := IsNormal.apply_omega0


@[simp]
theorem iSup_add_nat (o : Ordinal) : ⨆ n : ℕ, o + n = o + ω :=
  (isNormal_add_right o).apply_omega0


set_option linter.deprecated false in
@[deprecated iSup_add_nat (since := "2024-08-27")]
theorem sup_add_nat (o : Ordinal) : (sup fun n : ℕ => o + n) = o + ω :=
  (isNormal_add_right o).apply_omega0


@[simp]
theorem iSup_mul_nat (o : Ordinal) : ⨆ n : ℕ, o * n = o * ω := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (iSup fun n => HMul.hMul o ↑n) (HMul.hMul o Ordinal.omega0)
  -/
  rcases eq_zero_or_pos o with (rfl | ho)
    /-
      case inl
      ⊢ Eq (iSup fun n => HMul.hMul 0 ↑n) (HMul.hMul 0 Ordinal.omega0)
    -/
  · rw [zero_mul]
    /-
      case inl
      ⊢ Eq (iSup fun n => HMul.hMul 0 ↑n) 0
    -/
    exact iSup_eq_zero_iff.2 fun n => zero_mul (n : Ordinal)
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      ⊢ Eq (iSup fun n => HMul.hMul o ↑n) (HMul.hMul o Ordinal.omega0)
    -/
  · exact (isNormal_mul_right ho).apply_omega0
    /-
      🎉 no goals
    -/


set_option linter.deprecated false in
@[deprecated iSup_add_nat (since := "2024-08-27")]
theorem sup_mul_nat (o : Ordinal) : (sup fun n : ℕ => o * n) = o * ω := by
  /-
    o : Ordinal.{u_1}
    ⊢ Eq (Ordinal.sup fun n => HMul.hMul o ↑n) (HMul.hMul o Ordinal.omega0)
  -/
  rcases eq_zero_or_pos o with (rfl | ho)
    /-
      case inl
      ⊢ Eq (Ordinal.sup fun n => HMul.hMul 0 ↑n) (HMul.hMul 0 Ordinal.omega0)
    -/
  · rw [zero_mul]
    /-
      case inl
      ⊢ Eq (Ordinal.sup fun n => HMul.hMul 0 ↑n) 0
    -/
    exact sup_eq_zero_iff.2 fun n => zero_mul (n : Ordinal)
    /-
      🎉 no goals
    -/
    /-
      case inr
      o : Ordinal.{u_1}
      ho : LT.lt 0 o
      ⊢ Eq (Ordinal.sup fun n => HMul.hMul o ↑n) (HMul.hMul o Ordinal.omega0)
    -/
  · exact (mul_isNormal ho).apply_omega0
    /-
      🎉 no goals
    -/


@[simp]
theorem add_one_of_aleph0_le {c} (h : ℵ₀ ≤ c) : c + 1 = c := by
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ Eq (HAdd.hAdd c 1) c
  -/
  rw [add_comm, ← card_ord c, ← card_one, ← card_add, one_add_of_omega0_le]
  /-
    c : Cardinal.{u_1}
    h : LE.le Cardinal.aleph0 c
    ⊢ LE.le Ordinal.omega0 c.ord
  -/
  rwa [← ord_aleph0, ord_le_ord]
  /-
    🎉 no goals
  -/


theorem isLimit_ord {c} (co : ℵ₀ ≤ c) : (ord c).IsLimit := by
  /-
    c : Cardinal.{u_1}
    co : LE.le Cardinal.aleph0 c
    ⊢ c.ord.IsLimit
  -/
  rw [isLimit_iff, isSuccPrelimit_iff_succ_lt]
  /-
    c : Cardinal.{u_1}
    co : LE.le Cardinal.aleph0 c
    ⊢ And (Ne c.ord 0) (∀ (a : Ordinal.{u_1}), LT.lt a c.ord → LT.lt (Order.succ a …
  -/
  refine ⟨fun h => aleph0_ne_zero ?_, fun a => lt_imp_lt_of_le_imp_le fun h => ?_⟩
    /-
      case refine_1
      c : Cardinal.{u_1}
      co : LE.le Cardinal.aleph0 c
      h : Eq c.ord 0
      ⊢ Eq Cardinal.aleph0 0
    -/
  · rw [← Ordinal.le_zero, ord_le] at h
    /-
      case refine_1
      c : Cardinal.{u_1}
      co : LE.le Cardinal.aleph0 c
      h : LE.le c (Ordinal.card 0)
      ⊢ Eq Cardinal.aleph0 0
    -/
    simpa only [card_zero, nonpos_iff_eq_zero] using co.trans h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      c : Cardinal.{u_1}
      co : LE.le Cardinal.aleph0 c
      a : Ordinal.{u_1}
      h : LE.le c.ord (Order.succ a)
      ⊢ LE.le c.ord a
    -/
  · rw [ord_le] at h ⊢
    /-
      case refine_2
      c : Cardinal.{u_1}
      co : LE.le Cardinal.aleph0 c
      a : Ordinal.{u_1}
      h : LE.le c (Order.succ a).card
      ⊢ LE.le c a.card
    -/
    rwa [← @add_one_of_aleph0_le (card a), ← card_succ]
    /-
      case refine_2
      c : Cardinal.{u_1}
      co : LE.le Cardinal.aleph0 c
      a : Ordinal.{u_1}
      h : LE.le c (Order.succ a).card
      ⊢ LE.le Cardinal.aleph0 a.card
    -/
    rw [← ord_le, ← le_succ_of_isLimit, ord_le]
      /-
        case refine_2
        c : Cardinal.{u_1}
        co : LE.le Cardinal.aleph0 c
        a : Ordinal.{u_1}
        h : LE.le c (Order.succ a).card
        ⊢ LE.le Cardinal.aleph0 (Order.succ a).card
      -/
    · exact co.trans h
      /-
        🎉 no goals
      -/
      /-
        case refine_2.h
        c : Cardinal.{u_1}
        co : LE.le Cardinal.aleph0 c
        a : Ordinal.{u_1}
        h : LE.le c (Order.succ a).card
        ⊢ Cardinal.aleph0.ord.IsLimit
      -/
    · rw [ord_aleph0]
      /-
        case refine_2.h
        c : Cardinal.{u_1}
        co : LE.le Cardinal.aleph0 c
        a : Ordinal.{u_1}
        h : LE.le c (Order.succ a).card
        ⊢ Ordinal.omega0.IsLimit
      -/
      exact Ordinal.isLimit_omega0
      /-
        🎉 no goals
      -/


@[deprecated "No deprecation message was provided."  (since := "2024-10-14")]
alias ord_isLimit := isLimit_ord


theorem noMaxOrder {c} (h : ℵ₀ ≤ c) : NoMaxOrder c.ord.toType :=
  toType_noMax_of_succ_lt fun _ ↦ (isLimit_ord h).succ_lt


