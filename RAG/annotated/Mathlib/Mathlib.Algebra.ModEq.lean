/-- `a ≡ b [PMOD p]` means that `b` is congruent to `a` modulo `p`.

Equivalently (as shown in `Algebra.Order.ToIntervalMod`), `b` does not lie in the open interval
`(a, a + p)` modulo `p`, or `toIcoMod hp a` disagrees with `toIocMod hp a` at `b`, or
`toIcoDiv hp a` disagrees with `toIocDiv hp a` at `b`. -/
def ModEq (p a b : α) : Prop :=
  ∃ z : ℤ, b - a = z • p


@[inherit_doc]
notation:50 a " ≡ " b " [PMOD " p "]" => ModEq p a b


@[refl, simp]
theorem modEq_refl (a : α) : a ≡ a [PMOD p] :=
         /-
           α : Type u_1
           inst✝ : AddCommGroup α
           p a : α
           ⊢ Eq (HSub.hSub a a) (HSMul.hSMul 0 p)
         -/
  ⟨0, by simp⟩
         /-
           🎉 no goals
         -/


theorem modEq_rfl : a ≡ a [PMOD p] :=
  modEq_refl _


theorem modEq_comm : a ≡ b [PMOD p] ↔ b ≡ a [PMOD p] :=
                                              /-
                                                α : Type u_1
                                                inst✝ : AddCommGroup α
                                                p a b : α
                                                ⊢ Iff (Exists fun b_1 => Eq (HSub.hSub b a) (HSMul.hSMul ((Equiv.symm (Equiv.n …
                                              -/
  (Equiv.neg _).exists_congr_left.trans <| by simp [ModEq, ← neg_eq_iff_eq_neg]
                                              /-
                                                🎉 no goals
                                              -/


alias ⟨ModEq.symm, _⟩ := modEq_comm


@[trans]
theorem ModEq.trans : a ≡ b [PMOD p] → b ≡ c [PMOD p] → a ≡ c [PMOD p] := fun ⟨m, hm⟩ ⟨n, hn⟩ =>
             /-
               α : Type u_1
               inst✝ : AddCommGroup α
               p a b c : α
               x✝¹ : AddCommGroup.ModEq p a b
               x✝ : AddCommGroup.ModEq p b c
               m : Int
               hm : Eq (HSub.hSub b a) (HSMul.hSMul m p)
               n : Int
               hn : Eq (HSub.hSub c b) (HSMul.hSMul n p)
               ⊢ Eq (HSub.hSub c a) (HSMul.hSMul (HAdd.hAdd m n) p)
             -/
  ⟨m + n, by simp [add_smul, ← hm, ← hn]⟩
             /-
               🎉 no goals
             -/


instance : IsRefl _ (ModEq p) :=
  ⟨modEq_refl⟩


@[simp]
theorem neg_modEq_neg : -a ≡ -b [PMOD p] ↔ a ≡ b [PMOD p] :=
                         /-
                           α : Type u_1
                           inst✝ : AddCommGroup α
                           p a b : α
                           ⊢ Iff (AddCommGroup.ModEq p (Neg.neg b) (Neg.neg a)) (AddCommGroup.ModEq p a b)
                         -/
  modEq_comm.trans <| by simp [ModEq, neg_add_eq_sub]
                         /-
                           🎉 no goals
                         -/


alias ⟨ModEq.of_neg, ModEq.neg⟩ := neg_modEq_neg


@[simp]
theorem modEq_neg : a ≡ b [PMOD -p] ↔ a ≡ b [PMOD p] :=
                         /-
                           α : Type u_1
                           inst✝ : AddCommGroup α
                           p a b : α
                           ⊢ Iff (AddCommGroup.ModEq (Neg.neg p) b a) (AddCommGroup.ModEq p a b)
                         -/
  modEq_comm.trans <| by simp [ModEq, ← neg_eq_iff_eq_neg]
                         /-
                           🎉 no goals
                         -/


alias ⟨ModEq.of_neg', ModEq.neg'⟩ := modEq_neg


theorem modEq_sub (a b : α) : a ≡ b [PMOD b - a] :=
  ⟨1, (one_smul _ _).symm⟩


@[simp]
                                                  /-
                                                    α : Type u_1
                                                    inst✝ : AddCommGroup α
                                                    a b : α
                                                    ⊢ Iff (AddCommGroup.ModEq 0 a b) (Eq a b)
                                                  -/
theorem modEq_zero : a ≡ b [PMOD 0] ↔ a = b := by simp [ModEq, sub_eq_zero, eq_comm]
                                                  /-
                                                    🎉 no goals
                                                  -/


@[simp]
theorem self_modEq_zero : p ≡ 0 [PMOD p] :=
          /-
            α : Type u_1
            inst✝ : AddCommGroup α
            p : α
            ⊢ Eq (HSub.hSub 0 p) (HSMul.hSMul (-1) p)
          -/
  ⟨-1, by simp⟩
          /-
            🎉 no goals
          -/


@[simp]
theorem zsmul_modEq_zero (z : ℤ) : z • p ≡ 0 [PMOD p] :=
          /-
            α : Type u_1
            inst✝ : AddCommGroup α
            p : α
            z : Int
            ⊢ Eq (HSub.hSub 0 (HSMul.hSMul z p)) (HSMul.hSMul (Neg.neg z) p)
          -/
  ⟨-z, by simp⟩
          /-
            🎉 no goals
          -/


theorem add_zsmul_modEq (z : ℤ) : a + z • p ≡ a [PMOD p] :=
          /-
            α : Type u_1
            inst✝ : AddCommGroup α
            p a : α
            z : Int
            ⊢ Eq (HSub.hSub a (HAdd.hAdd a (HSMul.hSMul z p))) (HSMul.hSMul (Neg.neg z) p)
          -/
  ⟨-z, by simp⟩
          /-
            🎉 no goals
          -/


theorem zsmul_add_modEq (z : ℤ) : z • p + a ≡ a [PMOD p] :=
          /-
            α : Type u_1
            inst✝ : AddCommGroup α
            p a : α
            z : Int
            ⊢ Eq (HSub.hSub a (HAdd.hAdd (HSMul.hSMul z p) a)) (HSMul.hSMul (Neg.neg z) p)
          -/
  ⟨-z, by simp [← sub_sub]⟩
          /-
            🎉 no goals
          -/


theorem add_nsmul_modEq (n : ℕ) : a + n • p ≡ a [PMOD p] :=
          /-
            α : Type u_1
            inst✝ : AddCommGroup α
            p a : α
            n : Nat
            ⊢ Eq (HSub.hSub a (HAdd.hAdd a (HSMul.hSMul n p))) (HSMul.hSMul (Neg.neg ↑n) p)
          -/
  ⟨-n, by simp⟩
          /-
            🎉 no goals
          -/


theorem nsmul_add_modEq (n : ℕ) : n • p + a ≡ a [PMOD p] :=
          /-
            α : Type u_1
            inst✝ : AddCommGroup α
            p a : α
            n : Nat
            ⊢ Eq (HSub.hSub a (HAdd.hAdd (HSMul.hSMul n p) a)) (HSMul.hSMul (Neg.neg ↑n) p)
          -/
  ⟨-n, by simp [← sub_sub]⟩
          /-
            🎉 no goals
          -/


protected theorem add_zsmul (z : ℤ) : a ≡ b [PMOD p] → a + z • p ≡ b [PMOD p] :=
  (add_zsmul_modEq _).trans


protected theorem zsmul_add (z : ℤ) : a ≡ b [PMOD p] → z • p + a ≡ b [PMOD p] :=
  (zsmul_add_modEq _).trans


protected theorem add_nsmul (n : ℕ) : a ≡ b [PMOD p] → a + n • p ≡ b [PMOD p] :=
  (add_nsmul_modEq _).trans


protected theorem nsmul_add (n : ℕ) : a ≡ b [PMOD p] → n • p + a ≡ b [PMOD p] :=
  (nsmul_add_modEq _).trans


protected theorem of_zsmul : a ≡ b [PMOD z • p] → a ≡ b [PMOD p] := fun ⟨m, hm⟩ =>
             /-
               α : Type u_1
               inst✝ : AddCommGroup α
               p a b : α
               z : Int
               x✝ : AddCommGroup.ModEq (HSMul.hSMul z p) a b
               m : Int
               hm : Eq (HSub.hSub b a) (HSMul.hSMul m (HSMul.hSMul z p))
               ⊢ Eq (HSub.hSub b a) (HSMul.hSMul (HMul.hMul m z) p)
             -/
  ⟨m * z, by rwa [mul_smul]⟩
             /-
               🎉 no goals
             -/


protected theorem of_nsmul : a ≡ b [PMOD n • p] → a ≡ b [PMOD p] := fun ⟨m, hm⟩ =>
             /-
               α : Type u_1
               inst✝ : AddCommGroup α
               p a b : α
               n : Nat
               x✝ : AddCommGroup.ModEq (HSMul.hSMul n p) a b
               m : Int
               hm : Eq (HSub.hSub b a) (HSMul.hSMul m (HSMul.hSMul n p))
               ⊢ Eq (HSub.hSub b a) (HSMul.hSMul (HMul.hMul m ↑n) p)
             -/
  ⟨m * n, by rwa [mul_smul, natCast_zsmul]⟩
             /-
               🎉 no goals
             -/


protected theorem zsmul : a ≡ b [PMOD p] → z • a ≡ z • b [PMOD z • p] :=
                            /-
                              α : Type u_1
                              inst✝ : AddCommGroup α
                              p a b : α
                              z m : Int
                              hm : Eq (HSub.hSub b a) (HSMul.hSMul m p)
                              ⊢ Eq (HSub.hSub (HSMul.hSMul z b) (HSMul.hSMul z a)) (HSMul.hSMul m (HSMul.hSM …
                            -/
  Exists.imp fun m hm => by rw [← smul_sub, hm, smul_comm]
                            /-
                              🎉 no goals
                            -/


protected theorem nsmul : a ≡ b [PMOD p] → n • a ≡ n • b [PMOD n • p] :=
                            /-
                              α : Type u_1
                              inst✝ : AddCommGroup α
                              p a b : α
                              n : Nat
                              m : Int
                              hm : Eq (HSub.hSub b a) (HSMul.hSMul m p)
                              ⊢ Eq (HSub.hSub (HSMul.hSMul n b) (HSMul.hSMul n a)) (HSMul.hSMul m (HSMul.hSM …
                            -/
  Exists.imp fun m hm => by rw [← smul_sub, hm, smul_comm]
                            /-
                              🎉 no goals
                            -/


@[simp]
theorem zsmul_modEq_zsmul [NoZeroSMulDivisors ℤ α] (hn : z ≠ 0) :
    z • a ≡ z • b [PMOD z • p] ↔ a ≡ b [PMOD p] :=
                           /-
                             α : Type u_1
                             inst✝¹ : AddCommGroup α
                             p a b : α
                             z : Int
                             inst✝ : NoZeroSMulDivisors Int α
                             hn : Ne z 0
                             m : Int
                             ⊢ Iff (Eq (HSub.hSub (HSMul.hSMul z b) (HSMul.hSMul z a)) (HSMul.hSMul m (HSMu …
                           -/
  exists_congr fun m => by rw [← smul_sub, smul_comm, smul_right_inj hn]
                           /-
                             🎉 no goals
                           -/


@[simp]
theorem nsmul_modEq_nsmul [NoZeroSMulDivisors ℕ α] (hn : n ≠ 0) :
    n • a ≡ n • b [PMOD n • p] ↔ a ≡ b [PMOD p] :=
                           /-
                             α : Type u_1
                             inst✝¹ : AddCommGroup α
                             p a b : α
                             n : Nat
                             inst✝ : NoZeroSMulDivisors Nat α
                             hn : Ne n 0
                             m : Int
                             ⊢ Iff (Eq (HSub.hSub (HSMul.hSMul n b) (HSMul.hSMul n a)) (HSMul.hSMul m (HSMu …
                           -/
  exists_congr fun m => by rw [← smul_sub, smul_comm, smul_right_inj hn]
                           /-
                             🎉 no goals
                           -/


alias ⟨ModEq.zsmul_cancel, _⟩ := zsmul_modEq_zsmul


alias ⟨ModEq.nsmul_cancel, _⟩ := nsmul_modEq_nsmul


@[simp]
protected theorem add_iff_left :
    a₁ ≡ b₁ [PMOD p] → (a₁ + a₂ ≡ b₁ + b₂ [PMOD p] ↔ a₂ ≡ b₂ [PMOD p]) := fun ⟨m, hm⟩ =>
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : AddCommGroup α
                                                         p a₁ a₂ b₁ b₂ : α
                                                         x✝ : AddCommGroup.ModEq p a₁ b₁
                                                         m : Int
                                                         hm : Eq (HSub.hSub b₁ a₁) (HSMul.hSMul m p)
                                                         ⊢ Iff (Exists fun b => Eq (HSub.hSub (HAdd.hAdd b₁ b₂) (HAdd.hAdd a₁ a₂)) (HSM …
                                                       -/
  (Equiv.addLeft m).symm.exists_congr_left.trans <| by simp [add_sub_add_comm, hm, add_smul, ModEq]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
protected theorem add_iff_right :
    a₂ ≡ b₂ [PMOD p] → (a₁ + a₂ ≡ b₁ + b₂ [PMOD p] ↔ a₁ ≡ b₁ [PMOD p]) := fun ⟨m, hm⟩ =>
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : AddCommGroup α
                                                          p a₁ a₂ b₁ b₂ : α
                                                          x✝ : AddCommGroup.ModEq p a₂ b₂
                                                          m : Int
                                                          hm : Eq (HSub.hSub b₂ a₂) (HSMul.hSMul m p)
                                                          ⊢ Iff (Exists fun b => Eq (HSub.hSub (HAdd.hAdd b₁ b₂) (HAdd.hAdd a₁ a₂)) (HSM …
                                                        -/
  (Equiv.addRight m).symm.exists_congr_left.trans <| by simp [add_sub_add_comm, hm, add_smul, ModEq]
                                                        /-
                                                          🎉 no goals
                                                        -/


@[simp]
protected theorem sub_iff_left :
    a₁ ≡ b₁ [PMOD p] → (a₁ - a₂ ≡ b₁ - b₂ [PMOD p] ↔ a₂ ≡ b₂ [PMOD p]) := fun ⟨m, hm⟩ =>
                                                       /-
                                                         α : Type u_1
                                                         inst✝ : AddCommGroup α
                                                         p a₁ a₂ b₁ b₂ : α
                                                         x✝ : AddCommGroup.ModEq p a₁ b₁
                                                         m : Int
                                                         hm : Eq (HSub.hSub b₁ a₁) (HSMul.hSMul m p)
                                                         ⊢ Iff (Exists fun b => Eq (HSub.hSub (HSub.hSub b₁ b₂) (HSub.hSub a₁ a₂)) (HSM …
                                                       -/
  (Equiv.subLeft m).symm.exists_congr_left.trans <| by simp [sub_sub_sub_comm, hm, sub_smul, ModEq]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[simp]
protected theorem sub_iff_right :
    a₂ ≡ b₂ [PMOD p] → (a₁ - a₂ ≡ b₁ - b₂ [PMOD p] ↔ a₁ ≡ b₁ [PMOD p]) := fun ⟨m, hm⟩ =>
                                                        /-
                                                          α : Type u_1
                                                          inst✝ : AddCommGroup α
                                                          p a₁ a₂ b₁ b₂ : α
                                                          x✝ : AddCommGroup.ModEq p a₂ b₂
                                                          m : Int
                                                          hm : Eq (HSub.hSub b₂ a₂) (HSMul.hSMul m p)
                                                          ⊢ Iff (Exists fun b => Eq (HSub.hSub (HSub.hSub b₁ b₂) (HSub.hSub a₁ a₂)) (HSM …
                                                        -/
  (Equiv.subRight m).symm.exists_congr_left.trans <| by simp [sub_sub_sub_comm, hm, sub_smul, ModEq]
                                                        /-
                                                          🎉 no goals
                                                        -/


protected alias ⟨add_left_cancel, add⟩ := ModEq.add_iff_left


protected alias ⟨add_right_cancel, _⟩ := ModEq.add_iff_right


protected alias ⟨sub_left_cancel, sub⟩ := ModEq.sub_iff_left


protected alias ⟨sub_right_cancel, _⟩ := ModEq.sub_iff_right


protected theorem add_left (c : α) (h : a ≡ b [PMOD p]) : c + a ≡ c + b [PMOD p] :=
  modEq_rfl.add h


protected theorem sub_left (c : α) (h : a ≡ b [PMOD p]) : c - a ≡ c - b [PMOD p] :=
  modEq_rfl.sub h


protected theorem add_right (c : α) (h : a ≡ b [PMOD p]) : a + c ≡ b + c [PMOD p] :=
  h.add modEq_rfl


protected theorem sub_right (c : α) (h : a ≡ b [PMOD p]) : a - c ≡ b - c [PMOD p] :=
  h.sub modEq_rfl


protected theorem add_left_cancel' (c : α) : c + a ≡ c + b [PMOD p] → a ≡ b [PMOD p] :=
  modEq_rfl.add_left_cancel


protected theorem add_right_cancel' (c : α) : a + c ≡ b + c [PMOD p] → a ≡ b [PMOD p] :=
  modEq_rfl.add_right_cancel


protected theorem sub_left_cancel' (c : α) : c - a ≡ c - b [PMOD p] → a ≡ b [PMOD p] :=
  modEq_rfl.sub_left_cancel


protected theorem sub_right_cancel' (c : α) : a - c ≡ b - c [PMOD p] → a ≡ b [PMOD p] :=
  modEq_rfl.sub_right_cancel


theorem modEq_sub_iff_add_modEq' : a ≡ b - c [PMOD p] ↔ c + a ≡ b [PMOD p] := by
  /-
    α : Type u_1
    inst✝ : AddCommGroup α
    p a b c : α
    ⊢ Iff (AddCommGroup.ModEq p a (HSub.hSub b c)) (AddCommGroup.ModEq p (HAdd.hAd …
  -/
  simp [ModEq, sub_sub]
  /-
    🎉 no goals
  -/


theorem modEq_sub_iff_add_modEq : a ≡ b - c [PMOD p] ↔ a + c ≡ b [PMOD p] :=
                                       /-
                                         α : Type u_1
                                         inst✝ : AddCommGroup α
                                         p a b c : α
                                         ⊢ Iff (AddCommGroup.ModEq p (HAdd.hAdd c a) b) (AddCommGroup.ModEq p (HAdd.hAd …
                                       -/
  modEq_sub_iff_add_modEq'.trans <| by rw [add_comm]
                                       /-
                                         🎉 no goals
                                       -/


theorem sub_modEq_iff_modEq_add' : a - b ≡ c [PMOD p] ↔ a ≡ b + c [PMOD p] :=
  modEq_comm.trans <| modEq_sub_iff_add_modEq'.trans modEq_comm


theorem sub_modEq_iff_modEq_add : a - b ≡ c [PMOD p] ↔ a ≡ c + b [PMOD p] :=
  modEq_comm.trans <| modEq_sub_iff_add_modEq.trans modEq_comm


@[simp]
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : AddCommGroup α
                                                                     p a b : α
                                                                     ⊢ Iff (AddCommGroup.ModEq p (HSub.hSub a b) 0) (AddCommGroup.ModEq p a b)
                                                                   -/
theorem sub_modEq_zero : a - b ≡ 0 [PMOD p] ↔ a ≡ b [PMOD p] := by simp [sub_modEq_iff_modEq_add]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
                                                                   /-
                                                                     α : Type u_1
                                                                     inst✝ : AddCommGroup α
                                                                     p a b : α
                                                                     ⊢ Iff (AddCommGroup.ModEq p (HAdd.hAdd a b) a) (AddCommGroup.ModEq p b 0)
                                                                   -/
theorem add_modEq_left : a + b ≡ a [PMOD p] ↔ b ≡ 0 [PMOD p] := by simp [← modEq_sub_iff_add_modEq']
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


@[simp]
                                                                    /-
                                                                      α : Type u_1
                                                                      inst✝ : AddCommGroup α
                                                                      p a b : α
                                                                      ⊢ Iff (AddCommGroup.ModEq p (HAdd.hAdd a b) b) (AddCommGroup.ModEq p a 0)
                                                                    -/
theorem add_modEq_right : a + b ≡ b [PMOD p] ↔ a ≡ 0 [PMOD p] := by simp [← modEq_sub_iff_add_modEq]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem modEq_iff_eq_add_zsmul : a ≡ b [PMOD p] ↔ ∃ z : ℤ, b = a + z • p := by
  /-
    α : Type u_1
    inst✝ : AddCommGroup α
    p a b : α
    ⊢ Iff (AddCommGroup.ModEq p a b) (Exists fun z => Eq b (HAdd.hAdd a (HSMul.hSM …
  -/
  simp_rw [ModEq, sub_eq_iff_eq_add']
  /-
    🎉 no goals
  -/


theorem not_modEq_iff_ne_add_zsmul : ¬a ≡ b [PMOD p] ↔ ∀ z : ℤ, b ≠ a + z • p := by
  /-
    α : Type u_1
    inst✝ : AddCommGroup α
    p a b : α
    ⊢ Iff (Not (AddCommGroup.ModEq p a b)) (∀ (z : Int), Ne b (HAdd.hAdd a (HSMul. …
  -/
  rw [modEq_iff_eq_add_zsmul, not_exists]
  /-
    🎉 no goals
  -/


theorem modEq_iff_eq_mod_zmultiples : a ≡ b [PMOD p] ↔ (b : α ⧸ AddSubgroup.zmultiples p) = a := by
  simp_rw [modEq_iff_eq_add_zsmul, QuotientAddGroup.eq_iff_sub_mem, AddSubgroup.mem_zmultiples_iff,
    eq_sub_iff_add_eq', eq_comm]


theorem not_modEq_iff_ne_mod_zmultiples :
    ¬a ≡ b [PMOD p] ↔ (b : α ⧸ AddSubgroup.zmultiples p) ≠ a :=
  modEq_iff_eq_mod_zmultiples.not


@[simp]
theorem modEq_iff_int_modEq {a b z : ℤ} : a ≡ b [PMOD z] ↔ a ≡ b [ZMOD z] := by
  /-
    a b z : Int
    ⊢ Iff (AddCommGroup.ModEq z a b) (z.ModEq a b)
  -/
  simp [ModEq, dvd_iff_exists_eq_mul_left, Int.modEq_iff_dvd]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem intCast_modEq_intCast {a b z : ℤ} : a ≡ b [PMOD (z : α)] ↔ a ≡ b [PMOD z] := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroupWithOne α
    inst✝ : CharZero α
    a b z : Int
    ⊢ Iff (AddCommGroup.ModEq ↑z ↑a ↑b) (AddCommGroup.ModEq z a b)
  -/
  simp_rw [ModEq, ← Int.cast_mul_eq_zsmul_cast]
  /-
    α : Type u_1
    inst✝¹ : AddCommGroupWithOne α
    inst✝ : CharZero α
    a b z : Int
    ⊢ Iff (Exists fun z_1 => Eq (HSub.hSub ↑b ↑a) ↑(HMul.hMul z_1 z)) (Exists fun  …
  -/
  norm_cast
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
lemma intCast_modEq_intCast' {a b : ℤ} {n : ℕ} : a ≡ b [PMOD (n : α)] ↔ a ≡ b [PMOD (n : ℤ)] := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroupWithOne α
    inst✝ : CharZero α
    a b : Int
    n : Nat
    ⊢ Iff (AddCommGroup.ModEq ↑n ↑a ↑b) (AddCommGroup.ModEq (↑n) a b)
  -/
  simpa using intCast_modEq_intCast (α := α) (z := n)
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem natCast_modEq_natCast {a b n : ℕ} : a ≡ b [PMOD (n : α)] ↔ a ≡ b [MOD n] := by
  simp_rw [← Int.natCast_modEq_iff, ← modEq_iff_int_modEq, ← @intCast_modEq_intCast α,
    Int.cast_natCast]


alias ⟨ModEq.of_intCast, ModEq.intCast⟩ := intCast_modEq_intCast


alias ⟨_root_.Nat.ModEq.of_natCast, ModEq.natCast⟩ := natCast_modEq_natCast


@[simp] lemma div_modEq_div (hc : c ≠ 0) : a / c ≡ b / c [PMOD p] ↔ a ≡ b [PMOD (p * c)] := by
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    a b c p : α
    hc : Ne c 0
    ⊢ Iff (AddCommGroup.ModEq p (HDiv.hDiv a c) (HDiv.hDiv b c)) (AddCommGroup.Mod …
  -/
  simp [ModEq, ← sub_div, div_eq_iff hc, mul_assoc]
  /-
    🎉 no goals
  -/


@[simp] lemma mul_modEq_mul_right (hc : c ≠ 0) : a * c ≡ b * c [PMOD p] ↔ a ≡ b [PMOD (p / c)] := by
  /-
    α : Type u_1
    inst✝ : DivisionRing α
    a b c p : α
    hc : Ne c 0
    ⊢ Iff (AddCommGroup.ModEq p (HMul.hMul a c) (HMul.hMul b c)) (AddCommGroup.Mod …
  -/
  rw [div_eq_mul_inv, ← div_modEq_div (inv_ne_zero hc), div_inv_eq_mul, div_inv_eq_mul]
  /-
    🎉 no goals
  -/


@[simp] lemma mul_modEq_mul_left (hc : c ≠ 0) : c * a ≡ c * b [PMOD p] ↔ a ≡ b [PMOD (p / c)] := by
  /-
    α : Type u_1
    inst✝ : Field α
    a b c p : α
    hc : Ne c 0
    ⊢ Iff (AddCommGroup.ModEq p (HMul.hMul c a) (HMul.hMul c b)) (AddCommGroup.Mod …
  -/
  simp [mul_comm c, hc]
  /-
    🎉 no goals
  -/


