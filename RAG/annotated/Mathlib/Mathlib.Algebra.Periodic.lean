/-- A function `f` is said to be `Periodic` with period `c` if for all `x`, `f (x + c) = f x`. -/
@[simp]
def Periodic [Add α] (f : α → β) (c : α) : Prop :=
  ∀ x : α, f (x + c) = f x


protected theorem Periodic.funext [Add α] (h : Periodic f c) : (fun x => f (x + c)) = f :=
  funext h


protected theorem Periodic.comp [Add α] (h : Periodic f c) (g : β → γ) : Periodic (g ∘ f) c := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝ : Add α
    h : Function.Periodic f c
    g : β → γ
    ⊢ Function.Periodic (Function.comp g f) c
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem Periodic.comp_addHom [Add α] [Add γ] (h : Periodic f c) (g : AddHom γ α) (g_inv : α → γ)
    (hg : RightInverse g_inv g) : Periodic (f ∘ g) (g_inv c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝¹ : Add α
    inst✝ : Add γ
    h : Function.Periodic f c
    g : AddHom γ α
    g_inv : α → γ
    hg : Function.RightInverse g_inv ⇑g
    x : γ
    ⊢ Eq (Function.comp f (⇑g) (HAdd.hAdd x (g_inv c))) (Function.comp f (⇑g) x)
  -/
  simp only [hg c, h (g x), map_add, comp_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem Periodic.mul [Add α] [Mul β] (hf : Periodic f c) (hg : Periodic g c) :
                             /-
                               α : Type u_1
                               β : Type u_2
                               f g : α → β
                               c : α
                               inst✝¹ : Add α
                               inst✝ : Mul β
                               hf : Function.Periodic f c
                               hg : Function.Periodic g c
                               ⊢ Function.Periodic (HMul.hMul f g) c
                             -/
    Periodic (f * g) c := by simp_all
                             /-
                               🎉 no goals
                             -/


@[to_additive]
protected theorem Periodic.div [Add α] [Div β] (hf : Periodic f c) (hg : Periodic g c) :
                             /-
                               α : Type u_1
                               β : Type u_2
                               f g : α → β
                               c : α
                               inst✝¹ : Add α
                               inst✝ : Div β
                               hf : Function.Periodic f c
                               hg : Function.Periodic g c
                               ⊢ Function.Periodic (HDiv.hDiv f g) c
                             -/
    Periodic (f / g) c := by simp_all
                             /-
                               🎉 no goals
                             -/


@[to_additive]
theorem _root_.List.periodic_prod [Add α] [Monoid β] (l : List (α → β))
    (hl : ∀ f ∈ l, Periodic f c) : Periodic l.prod c := by
  induction l with
  | nil => simp
  | cons g l ih =>
    rw [List.forall_mem_cons] at hl
    simpa only [List.prod_cons] using hl.1.mul (ih hl.2)


@[to_additive]
theorem _root_.Multiset.periodic_prod [Add α] [CommMonoid β] (s : Multiset (α → β))
    (hs : ∀ f ∈ s, Periodic f c) : Periodic s.prod c :=
  (s.prod_toList ▸ s.toList.periodic_prod) fun f hf => hs f <| Multiset.mem_toList.mp hf


@[to_additive]
theorem _root_.Finset.periodic_prod [Add α] [CommMonoid β] {ι : Type*} {f : ι → α → β}
    (s : Finset ι) (hs : ∀ i ∈ s, Periodic (f i) c) : Periodic (∏ i ∈ s, f i) c :=
                                                        /-
                                                          α : Type u_1
                                                          β : Type u_2
                                                          c : α
                                                          inst✝¹ : Add α
                                                          inst✝ : CommMonoid β
                                                          ι : Type u_4
                                                          f : ι → α → β
                                                          s : Finset ι
                                                          hs : ∀ (i : ι), Membership.mem s i → Function.Periodic (f i) c
                                                          ⊢ ∀ (f_1 : α → β), Membership.mem (List.map f s.toList) f_1 → Function.Periodi …
                                                        -/
  s.prod_to_list f ▸ (s.toList.map f).periodic_prod (by simpa [-Periodic] )
                                                        /-
                                                          🎉 no goals
                                                        -/


@[to_additive]
protected theorem Periodic.smul [Add α] [SMul γ β] (h : Periodic f c) (a : γ) :
                             /-
                               α : Type u_1
                               β : Type u_2
                               γ : Type u_3
                               f : α → β
                               c : α
                               inst✝¹ : Add α
                               inst✝ : SMul γ β
                               h : Function.Periodic f c
                               a : γ
                               ⊢ Function.Periodic (HSMul.hSMul a f) c
                             -/
    Periodic (a • f) c := by simp_all
                             /-
                               🎉 no goals
                             -/


protected theorem Periodic.const_smul [AddMonoid α] [Group γ] [DistribMulAction γ α]
    (h : Periodic f c) (a : γ) : Periodic (fun x => f (a • x)) (a⁻¹ • c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝² : AddMonoid α
    inst✝¹ : Group γ
    inst✝ : DistribMulAction γ α
    h : Function.Periodic f c
    a : γ
    x : α
    ⊢ Eq ((fun x => f (HSMul.hSMul a x)) (HAdd.hAdd x (HSMul.hSMul (Inv.inv a) c)) …
  -/
  simpa only [smul_add, smul_inv_smul] using h (a • x)
  /-
    🎉 no goals
  -/


protected theorem Periodic.const_smul₀ [AddCommMonoid α] [DivisionSemiring γ] [Module γ α]
    (h : Periodic f c) (a : γ) : Periodic (fun x => f (a • x)) (a⁻¹ • c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝² : AddCommMonoid α
    inst✝¹ : DivisionSemiring γ
    inst✝ : Module γ α
    h : Function.Periodic f c
    a : γ
    x : α
    ⊢ Eq ((fun x => f (HSMul.hSMul a x)) (HAdd.hAdd x (HSMul.hSMul (Inv.inv a) c)) …
  -/
  by_cases ha : a = 0
    /-
      case pos
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β
      c : α
      inst✝² : AddCommMonoid α
      inst✝¹ : DivisionSemiring γ
      inst✝ : Module γ α
      h : Function.Periodic f c
      a : γ
      x : α
      ha : Eq a 0
      ⊢ Eq ((fun x => f (HSMul.hSMul a x)) (HAdd.hAdd x (HSMul.hSMul (Inv.inv a) c)) …
    -/
  · simp only [ha, zero_smul]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f : α → β
      c : α
      inst✝² : AddCommMonoid α
      inst✝¹ : DivisionSemiring γ
      inst✝ : Module γ α
      h : Function.Periodic f c
      a : γ
      x : α
      ha : Not (Eq a 0)
      ⊢ Eq ((fun x => f (HSMul.hSMul a x)) (HAdd.hAdd x (HSMul.hSMul (Inv.inv a) c)) …
    -/
  · simpa only [smul_add, smul_inv_smul₀ ha] using h (a • x)
    /-
      🎉 no goals
    -/


protected theorem Periodic.const_mul [DivisionSemiring α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (a * x)) (a⁻¹ * c) :=
  Periodic.const_smul₀ h a


theorem Periodic.const_inv_smul [AddMonoid α] [Group γ] [DistribMulAction γ α] (h : Periodic f c)
    (a : γ) : Periodic (fun x => f (a⁻¹ • x)) (a • c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝² : AddMonoid α
    inst✝¹ : Group γ
    inst✝ : DistribMulAction γ α
    h : Function.Periodic f c
    a : γ
    ⊢ Function.Periodic (fun x => f (HSMul.hSMul (Inv.inv a) x)) (HSMul.hSMul a c)
  -/
  simpa only [inv_inv] using h.const_smul a⁻¹
  /-
    🎉 no goals
  -/


theorem Periodic.const_inv_smul₀ [AddCommMonoid α] [DivisionSemiring γ] [Module γ α]
    (h : Periodic f c) (a : γ) : Periodic (fun x => f (a⁻¹ • x)) (a • c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝² : AddCommMonoid α
    inst✝¹ : DivisionSemiring γ
    inst✝ : Module γ α
    h : Function.Periodic f c
    a : γ
    ⊢ Function.Periodic (fun x => f (HSMul.hSMul (Inv.inv a) x)) (HSMul.hSMul a c)
  -/
  simpa only [inv_inv] using h.const_smul₀ a⁻¹
  /-
    🎉 no goals
  -/


theorem Periodic.const_inv_mul [DivisionSemiring α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (a⁻¹ * x)) (a * c) :=
  h.const_inv_smul₀ a


theorem Periodic.mul_const [DivisionSemiring α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (x * a)) (c * a⁻¹) :=
  h.const_smul₀ (MulOpposite.op a)


theorem Periodic.mul_const' [DivisionSemiring α] (h : Periodic f c) (a : α) :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  f : α → β
                                                  c : α
                                                  inst✝ : DivisionSemiring α
                                                  h : Function.Periodic f c
                                                  a : α
                                                  ⊢ Function.Periodic (fun x => f (HMul.hMul x a)) (HDiv.hDiv c a)
                                                -/
    Periodic (fun x => f (x * a)) (c / a) := by simpa only [div_eq_mul_inv] using h.mul_const a
                                                /-
                                                  🎉 no goals
                                                -/


theorem Periodic.mul_const_inv [DivisionSemiring α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (x * a⁻¹)) (c * a) :=
  h.const_inv_smul₀ (MulOpposite.op a)


theorem Periodic.div_const [DivisionSemiring α] (h : Periodic f c) (a : α) :
                                                /-
                                                  α : Type u_1
                                                  β : Type u_2
                                                  f : α → β
                                                  c : α
                                                  inst✝ : DivisionSemiring α
                                                  h : Function.Periodic f c
                                                  a : α
                                                  ⊢ Function.Periodic (fun x => f (HDiv.hDiv x a)) (HMul.hMul c a)
                                                -/
    Periodic (fun x => f (x / a)) (c * a) := by simpa only [div_eq_mul_inv] using h.mul_const_inv a
                                                /-
                                                  🎉 no goals
                                                -/


theorem Periodic.add_period [AddSemigroup α] (h1 : Periodic f c₁) (h2 : Periodic f c₂) :
                               /-
                                 α : Type u_1
                                 β : Type u_2
                                 f : α → β
                                 c₁ c₂ : α
                                 inst✝ : AddSemigroup α
                                 h1 : Function.Periodic f c₁
                                 h2 : Function.Periodic f c₂
                                 ⊢ Function.Periodic f (HAdd.hAdd c₁ c₂)
                               -/
    Periodic f (c₁ + c₂) := by simp_all [← add_assoc]
                               /-
                                 🎉 no goals
                               -/


theorem Periodic.sub_eq [AddGroup α] (h : Periodic f c) (x : α) : f (x - c) = f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddGroup α
    h : Function.Periodic f c
    x : α
    ⊢ Eq (f (HSub.hSub x c)) (f x)
  -/
  simpa only [sub_add_cancel] using (h (x - c)).symm
  /-
    🎉 no goals
  -/


theorem Periodic.sub_eq' [AddCommGroup α] (h : Periodic f c) : f (c - x) = f (-x) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝ : AddCommGroup α
    h : Function.Periodic f c
    ⊢ Eq (f (HSub.hSub c x)) (f (Neg.neg x))
  -/
  simpa only [sub_eq_neg_add] using h (-x)
  /-
    🎉 no goals
  -/


protected theorem Periodic.neg [AddGroup α] (h : Periodic f c) : Periodic f (-c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddGroup α
    h : Function.Periodic f c
    ⊢ Function.Periodic f (Neg.neg c)
  -/
  simpa only [sub_eq_add_neg, Periodic] using h.sub_eq
  /-
    🎉 no goals
  -/


theorem Periodic.sub_period [AddGroup α] (h1 : Periodic f c₁) (h2 : Periodic f c₂) :
    Periodic f (c₁ - c₂) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c₁ c₂ : α
    inst✝ : AddGroup α
    h1 : Function.Periodic f c₁
    h2 : Function.Periodic f c₂
    x : α
    ⊢ Eq (f (HAdd.hAdd x (HSub.hSub c₁ c₂))) (f x)
  -/
  rw [sub_eq_add_neg, ← add_assoc, h2.neg, h1]
  /-
    🎉 no goals
  -/


theorem Periodic.const_add [AddSemigroup α] (h : Periodic f c) (a : α) :
                                                   /-
                                                     α : Type u_1
                                                     β : Type u_2
                                                     f : α → β
                                                     c : α
                                                     inst✝ : AddSemigroup α
                                                     h : Function.Periodic f c
                                                     a x : α
                                                     ⊢ Eq ((fun x => f (HAdd.hAdd a x)) (HAdd.hAdd x c)) ((fun x => f (HAdd.hAdd a  …
                                                   -/
    Periodic (fun x => f (a + x)) c := fun x => by simpa [add_assoc] using h (a + x)
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem Periodic.add_const [AddCommSemigroup α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (x + a)) c := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommSemigroup α
    h : Function.Periodic f c
    a x : α
    ⊢ Eq ((fun x => f (HAdd.hAdd x a)) (HAdd.hAdd x c)) ((fun x => f (HAdd.hAdd x  …
  -/
  simpa only [add_right_comm] using h (x + a)
  /-
    🎉 no goals
  -/


theorem Periodic.const_sub [AddCommGroup α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (a - x)) c := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommGroup α
    h : Function.Periodic f c
    a x : α
    ⊢ Eq ((fun x => f (HSub.hSub a x)) (HAdd.hAdd x c)) ((fun x => f (HSub.hSub a  …
  -/
  simp only [← sub_sub, h.sub_eq]
  /-
    🎉 no goals
  -/


theorem Periodic.sub_const [AddCommGroup α] (h : Periodic f c) (a : α) :
    Periodic (fun x => f (x - a)) c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommGroup α
    h : Function.Periodic f c
    a : α
    ⊢ Function.Periodic (fun x => f (HSub.hSub x a)) c
  -/
  simpa only [sub_eq_add_neg] using h.add_const (-a)
  /-
    🎉 no goals
  -/


theorem Periodic.nsmul [AddMonoid α] (h : Periodic f c) (n : ℕ) : Periodic f (n • c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddMonoid α
    h : Function.Periodic f c
    n : Nat
    ⊢ Function.Periodic f (HSMul.hSMul n c)
  -/
                  /-
                    🎉 no goals
                  -/
  induction n <;> simp_all [add_nsmul, ← add_assoc, zero_nsmul]
                  /-
                    🎉 no goals
                  -/


theorem Periodic.nat_mul [Semiring α] (h : Periodic f c) (n : ℕ) : Periodic f (n * c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : Semiring α
    h : Function.Periodic f c
    n : Nat
    ⊢ Function.Periodic f (HMul.hMul (↑n) c)
  -/
  simpa only [nsmul_eq_mul] using h.nsmul n
  /-
    🎉 no goals
  -/


theorem Periodic.neg_nsmul [AddGroup α] (h : Periodic f c) (n : ℕ) : Periodic f (-(n • c)) :=
  (h.nsmul n).neg


theorem Periodic.neg_nat_mul [Ring α] (h : Periodic f c) (n : ℕ) : Periodic f (-(n * c)) :=
  (h.nat_mul n).neg


theorem Periodic.sub_nsmul_eq [AddGroup α] (h : Periodic f c) (n : ℕ) : f (x - n • c) = f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝ : AddGroup α
    h : Function.Periodic f c
    n : Nat
    ⊢ Eq (f (HSub.hSub x (HSMul.hSMul n c))) (f x)
  -/
  simpa only [sub_eq_add_neg] using h.neg_nsmul n x
  /-
    🎉 no goals
  -/


theorem Periodic.sub_nat_mul_eq [Ring α] (h : Periodic f c) (n : ℕ) : f (x - n * c) = f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝ : Ring α
    h : Function.Periodic f c
    n : Nat
    ⊢ Eq (f (HSub.hSub x (HMul.hMul (↑n) c))) (f x)
  -/
  simpa only [nsmul_eq_mul] using h.sub_nsmul_eq n
  /-
    🎉 no goals
  -/


theorem Periodic.nsmul_sub_eq [AddCommGroup α] (h : Periodic f c) (n : ℕ) :
    f (n • c - x) = f (-x) :=
  (h.nsmul n).sub_eq'


theorem Periodic.nat_mul_sub_eq [Ring α] (h : Periodic f c) (n : ℕ) : f (n * c - x) = f (-x) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝ : Ring α
    h : Function.Periodic f c
    n : Nat
    ⊢ Eq (f (HSub.hSub (HMul.hMul (↑n) c) x)) (f (Neg.neg x))
  -/
  simpa only [sub_eq_neg_add] using h.nat_mul n (-x)
  /-
    🎉 no goals
  -/


protected theorem Periodic.zsmul [AddGroup α] (h : Periodic f c) (n : ℤ) : Periodic f (n • c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddGroup α
    h : Function.Periodic f c
    n : Int
    ⊢ Function.Periodic f (HSMul.hSMul n c)
  -/
  rcases n with n | n
    /-
      case ofNat
      α : Type u_1
      β : Type u_2
      f : α → β
      c : α
      inst✝ : AddGroup α
      h : Function.Periodic f c
      n : Nat
      ⊢ Function.Periodic f (HSMul.hSMul (Int.ofNat n) c)
    -/
  · simpa only [Int.ofNat_eq_coe, natCast_zsmul] using h.nsmul n
    /-
      🎉 no goals
    -/
    /-
      case negSucc
      α : Type u_1
      β : Type u_2
      f : α → β
      c : α
      inst✝ : AddGroup α
      h : Function.Periodic f c
      n : Nat
      ⊢ Function.Periodic f (HSMul.hSMul (Int.negSucc n) c)
    -/
  · simpa only [negSucc_zsmul] using (h.nsmul (n + 1)).neg
    /-
      🎉 no goals
    -/


protected theorem Periodic.int_mul [Ring α] (h : Periodic f c) (n : ℤ) : Periodic f (n * c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : Ring α
    h : Function.Periodic f c
    n : Int
    ⊢ Function.Periodic f (HMul.hMul (↑n) c)
  -/
  simpa only [zsmul_eq_mul] using h.zsmul n
  /-
    🎉 no goals
  -/


theorem Periodic.sub_zsmul_eq [AddGroup α] (h : Periodic f c) (n : ℤ) : f (x - n • c) = f x :=
  (h.zsmul n).sub_eq x


theorem Periodic.sub_int_mul_eq [Ring α] (h : Periodic f c) (n : ℤ) : f (x - n * c) = f x :=
  (h.int_mul n).sub_eq x


theorem Periodic.zsmul_sub_eq [AddCommGroup α] (h : Periodic f c) (n : ℤ) :
    f (n • c - x) = f (-x) :=
  (h.zsmul _).sub_eq'


theorem Periodic.int_mul_sub_eq [Ring α] (h : Periodic f c) (n : ℤ) : f (n * c - x) = f (-x) :=
  (h.int_mul _).sub_eq'


protected theorem Periodic.eq [AddZeroClass α] (h : Periodic f c) : f c = f 0 := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddZeroClass α
    h : Function.Periodic f c
    ⊢ Eq (f c) (f 0)
  -/
  simpa only [zero_add] using h 0
  /-
    🎉 no goals
  -/


protected theorem Periodic.neg_eq [AddGroup α] (h : Periodic f c) : f (-c) = f 0 :=
  h.neg.eq


protected theorem Periodic.nsmul_eq [AddMonoid α] (h : Periodic f c) (n : ℕ) : f (n • c) = f 0 :=
  (h.nsmul n).eq


theorem Periodic.nat_mul_eq [Semiring α] (h : Periodic f c) (n : ℕ) : f (n * c) = f 0 :=
  (h.nat_mul n).eq


theorem Periodic.zsmul_eq [AddGroup α] (h : Periodic f c) (n : ℤ) : f (n • c) = f 0 :=
  (h.zsmul n).eq


theorem Periodic.int_mul_eq [Ring α] (h : Periodic f c) (n : ℤ) : f (n * c) = f 0 :=
  (h.int_mul n).eq


/-- If a function `f` is `Periodic` with positive period `c`, then for all `x` there exists some
  `y ∈ Ico 0 c` such that `f x = f y`. -/
theorem Periodic.exists_mem_Ico₀ [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c)
    (hc : 0 < c) (x) : ∃ y ∈ Ico 0 c, f x = f y :=
  let ⟨n, H, _⟩ := existsUnique_zsmul_near_of_pos' hc x
  ⟨x - n • c, H, (h.sub_zsmul_eq n).symm⟩


/-- If a function `f` is `Periodic` with positive period `c`, then for all `x` there exists some
  `y ∈ Ico a (a + c)` such that `f x = f y`. -/
theorem Periodic.exists_mem_Ico [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c)
    (hc : 0 < c) (x a) : ∃ y ∈ Ico a (a + c), f x = f y :=
  let ⟨n, H, _⟩ := existsUnique_add_zsmul_mem_Ico hc x a
  ⟨x + n • c, H, (h.zsmul n x).symm⟩


/-- If a function `f` is `Periodic` with positive period `c`, then for all `x` there exists some
  `y ∈ Ioc a (a + c)` such that `f x = f y`. -/
theorem Periodic.exists_mem_Ioc [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c)
    (hc : 0 < c) (x a) : ∃ y ∈ Ioc a (a + c), f x = f y :=
  let ⟨n, H, _⟩ := existsUnique_add_zsmul_mem_Ioc hc x a
  ⟨x + n • c, H, (h.zsmul n x).symm⟩


theorem Periodic.image_Ioc [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c)
    (hc : 0 < c) (a : α) : f '' Ioc a (a + c) = range f :=
  (image_subset_range _ _).antisymm <| range_subset_iff.2 fun x =>
    let ⟨y, hy, hyx⟩ := h.exists_mem_Ioc hc x a
    ⟨y, hy, hyx.symm⟩


theorem Periodic.image_Icc [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c)
    (hc : 0 < c) (a : α) : f '' Icc a (a + c) = range f :=
  (image_subset_range _ _).antisymm <| h.image_Ioc hc a ▸ image_subset _ Ioc_subset_Icc_self


theorem Periodic.image_uIcc [LinearOrderedAddCommGroup α] [Archimedean α] (h : Periodic f c)
    (hc : c ≠ 0) (a : α) : f '' uIcc a (a + c) = range f := by
  cases hc.lt_or_lt with
  | inl hc =>
    rw [uIcc_of_ge (add_le_of_nonpos_right hc.le), ← h.neg.image_Icc (neg_pos.2 hc) (a + c),
      add_neg_cancel_right]
  | inr hc => rw [uIcc_of_le (le_add_of_nonneg_right hc.le), h.image_Icc hc]


theorem periodic_with_period_zero [AddZeroClass α] (f : α → β) : Periodic f 0 := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    inst✝ : AddZeroClass α
    f : α → β
    x : α
    ⊢ Eq (f (HAdd.hAdd x 0)) (f x)
  -/
  rw [add_zero]
  /-
    🎉 no goals
  -/


theorem Periodic.map_vadd_zmultiples [AddCommGroup α] (hf : Periodic f c)
    (a : AddSubgroup.zmultiples c) (x : α) : f (a +ᵥ x) = f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommGroup α
    hf : Function.Periodic f c
    a : Subtype fun x => Membership.mem (AddSubgroup.zmultiples c) x
    x : α
    ⊢ Eq (f (HVAdd.hVAdd a x)) (f x)
  -/
  rcases a with ⟨_, m, rfl⟩
  /-
    case mk.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommGroup α
    hf : Function.Periodic f c
    x : α
    m : Int
    ⊢ Eq (f (HVAdd.hVAdd ⟨(fun x => HSMul.hSMul x c) m, ⋯⟩ x)) (f x)
  -/
  simp [AddSubgroup.vadd_def, add_comm _ x, hf.zsmul m x]
  /-
    🎉 no goals
  -/


theorem Periodic.map_vadd_multiples [AddCommMonoid α] (hf : Periodic f c)
    (a : AddSubmonoid.multiples c) (x : α) : f (a +ᵥ x) = f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommMonoid α
    hf : Function.Periodic f c
    a : Subtype fun x => Membership.mem (AddSubmonoid.multiples c) x
    x : α
    ⊢ Eq (f (HVAdd.hVAdd a x)) (f x)
  -/
  rcases a with ⟨_, m, rfl⟩
  /-
    case mk.intro
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝ : AddCommMonoid α
    hf : Function.Periodic f c
    x : α
    m : Nat
    ⊢ Eq (f (HVAdd.hVAdd ⟨(fun i => HSMul.hSMul i c) m, ⋯⟩ x)) (f x)
  -/
  simp [AddSubmonoid.vadd_def, add_comm _ x, hf.nsmul m x]
  /-
    🎉 no goals
  -/


/-- Lift a periodic function to a function from the quotient group. -/
def Periodic.lift [AddGroup α] (h : Periodic f c) (x : α ⧸ AddSubgroup.zmultiples c) : β :=
  Quotient.liftOn' x f fun a b h' => by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f g : α → β
      c c₁ c₂ x✝ : α
      inst✝ : AddGroup α
      h : Function.Periodic f c
      x : HasQuotient.Quotient α (AddSubgroup.zmultiples c)
      a b : α
      h' : (QuotientAddGroup.leftRel (AddSubgroup.zmultiples c)) a b
      ⊢ Eq (f a) (f b)
    -/
    rw [QuotientAddGroup.leftRel_apply] at h'
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f g : α → β
      c c₁ c₂ x✝ : α
      inst✝ : AddGroup α
      h : Function.Periodic f c
      x : HasQuotient.Quotient α (AddSubgroup.zmultiples c)
      a b : α
      h' : Membership.mem (AddSubgroup.zmultiples c) (HAdd.hAdd (Neg.neg a) b)
      ⊢ Eq (f a) (f b)
    -/
    obtain ⟨k, hk⟩ := h'
    /-
      case intro
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      f g : α → β
      c c₁ c₂ x✝ : α
      inst✝ : AddGroup α
      h : Function.Periodic f c
      x : HasQuotient.Quotient α (AddSubgroup.zmultiples c)
      a b : α
      k : Int
      hk : Eq ((fun x => HSMul.hSMul x c) k) (HAdd.hAdd (Neg.neg a) b)
      ⊢ Eq (f a) (f b)
    -/
    exact (h.zsmul k _).symm.trans (congr_arg f (add_eq_of_eq_neg_add hk))
    /-
      🎉 no goals
    -/


@[simp]
theorem Periodic.lift_coe [AddGroup α] (h : Periodic f c) (a : α) :
    h.lift (a : α ⧸ AddSubgroup.zmultiples c) = f a :=
  rfl


/-- A periodic function `f : R → X` on a semiring (or, more generally, `AddZeroClass`)
of non-zero period is not injective. -/
lemma Periodic.not_injective {R X : Type*} [AddZeroClass R] {f : R → X} {c : R}
    (hf : Periodic f c) (hc : c ≠ 0) : ¬ Injective f := fun h ↦ hc <| h hf.eq


/-- A function `f` is said to be `antiperiodic` with antiperiod `c` if for all `x`,
  `f (x + c) = -f x`. -/
@[simp]
def Antiperiodic [Add α] [Neg β] (f : α → β) (c : α) : Prop :=
  ∀ x : α, f (x + c) = -f x


protected theorem Antiperiodic.funext [Add α] [Neg β] (h : Antiperiodic f c) :
    (fun x => f (x + c)) = -f :=
  funext h


protected theorem Antiperiodic.funext' [Add α] [InvolutiveNeg β] (h : Antiperiodic f c) :
    (fun x => -f (x + c)) = f :=
  neg_eq_iff_eq_neg.mpr h.funext


/-- If a function is `antiperiodic` with antiperiod `c`, then it is also `Periodic` with period
`2 • c`. -/
protected theorem Antiperiodic.periodic [AddMonoid α] [InvolutiveNeg β]
                                                      /-
                                                        α : Type u_1
                                                        β : Type u_2
                                                        f : α → β
                                                        c : α
                                                        inst✝¹ : AddMonoid α
                                                        inst✝ : InvolutiveNeg β
                                                        h : Function.Antiperiodic f c
                                                        ⊢ Function.Periodic f (HSMul.hSMul 2 c)
                                                      -/
    (h : Antiperiodic f c) : Periodic f (2 • c) := by simp [two_nsmul, ← add_assoc, h _]
                                                      /-
                                                        🎉 no goals
                                                      -/


/-- If a function is `antiperiodic` with antiperiod `c`, then it is also `Periodic` with period
  `2 * c`. -/
protected theorem Antiperiodic.periodic_two_mul [Semiring α] [InvolutiveNeg β]
    (h : Antiperiodic f c) : Periodic f (2 * c) := nsmul_eq_mul 2 c ▸ h.periodic


protected theorem Antiperiodic.eq [AddZeroClass α] [Neg β] (h : Antiperiodic f c) : f c = -f 0 := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddZeroClass α
    inst✝ : Neg β
    h : Function.Antiperiodic f c
    ⊢ Eq (f c) (Neg.neg (f 0))
  -/
  simpa only [zero_add] using h 0
  /-
    🎉 no goals
  -/


theorem Antiperiodic.even_nsmul_periodic [AddMonoid α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℕ) : Periodic f ((2 * n) • c) := mul_nsmul c 2 n ▸ h.periodic.nsmul n


theorem Antiperiodic.nat_even_mul_periodic [Semiring α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℕ) : Periodic f (n * (2 * c)) :=
  h.periodic_two_mul.nat_mul n


theorem Antiperiodic.odd_nsmul_antiperiodic [AddMonoid α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℕ) : Antiperiodic f ((2 * n + 1) • c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddMonoid α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Nat
    x : α
    ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul (HAdd.hAdd (HMul.hMul 2 n) 1) c))) (Neg.neg  …
  -/
  rw [add_nsmul, one_nsmul, ← add_assoc, h, h.even_nsmul_periodic]
  /-
    🎉 no goals
  -/


theorem Antiperiodic.nat_odd_mul_antiperiodic [Semiring α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℕ) : Antiperiodic f (n * (2 * c) + c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : Semiring α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Nat
    x : α
    ⊢ Eq (f (HAdd.hAdd x (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul 2 c)) c))) (Neg.neg …
  -/
  rw [← add_assoc, h, h.nat_even_mul_periodic]
  /-
    🎉 no goals
  -/


theorem Antiperiodic.even_zsmul_periodic [AddGroup α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℤ) : Periodic f ((2 * n) • c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Function.Periodic f (HSMul.hSMul (HMul.hMul 2 n) c)
  -/
  rw [mul_comm, mul_zsmul, two_zsmul, ← two_nsmul]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Function.Periodic f (HSMul.hSMul n (HSMul.hSMul 2 c))
  -/
  exact h.periodic.zsmul n
  /-
    🎉 no goals
  -/


theorem Antiperiodic.int_even_mul_periodic [Ring α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℤ) : Periodic f (n * (2 * c)) :=
  h.periodic_two_mul.int_mul n


theorem Antiperiodic.odd_zsmul_antiperiodic [AddGroup α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℤ) : Antiperiodic f ((2 * n + 1) • c) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Function.Antiperiodic f (HSMul.hSMul (HAdd.hAdd (HMul.hMul 2 n) 1) c)
  -/
  intro x
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Int
    x : α
    ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul (HAdd.hAdd (HMul.hMul 2 n) 1) c))) (Neg.neg  …
  -/
  rw [add_zsmul, one_zsmul, ← add_assoc, h, h.even_zsmul_periodic]
  /-
    🎉 no goals
  -/


theorem Antiperiodic.int_odd_mul_antiperiodic [Ring α] [InvolutiveNeg β] (h : Antiperiodic f c)
    (n : ℤ) : Antiperiodic f (n * (2 * c) + c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : Ring α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    n : Int
    x : α
    ⊢ Eq (f (HAdd.hAdd x (HAdd.hAdd (HMul.hMul (↑n) (HMul.hMul 2 c)) c))) (Neg.neg …
  -/
  rw [← add_assoc, h, h.int_even_mul_periodic]
  /-
    🎉 no goals
  -/


theorem Antiperiodic.sub_eq [AddGroup α] [InvolutiveNeg β] (h : Antiperiodic f c) (x : α) :
                           /-
                             α : Type u_1
                             β : Type u_2
                             f : α → β
                             c : α
                             inst✝¹ : AddGroup α
                             inst✝ : InvolutiveNeg β
                             h : Function.Antiperiodic f c
                             x : α
                             ⊢ Eq (f (HSub.hSub x c)) (Neg.neg (f x))
                           -/
    f (x - c) = -f x := by simp only [← neg_eq_iff_eq_neg, ← h (x - c), sub_add_cancel]
                           /-
                             🎉 no goals
                           -/


theorem Antiperiodic.sub_eq' [AddCommGroup α] [Neg β] (h : Antiperiodic f c) :
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                c x : α
                                inst✝¹ : AddCommGroup α
                                inst✝ : Neg β
                                h : Function.Antiperiodic f c
                                ⊢ Eq (f (HSub.hSub c x)) (Neg.neg (f (Neg.neg x)))
                              -/
    f (c - x) = -f (-x) := by simpa only [sub_eq_neg_add] using h (-x)
                              /-
                                🎉 no goals
                              -/


protected theorem Antiperiodic.neg [AddGroup α] [InvolutiveNeg β] (h : Antiperiodic f c) :
                              /-
                                α : Type u_1
                                β : Type u_2
                                f : α → β
                                c : α
                                inst✝¹ : AddGroup α
                                inst✝ : InvolutiveNeg β
                                h : Function.Antiperiodic f c
                                ⊢ Function.Antiperiodic f (Neg.neg c)
                              -/
    Antiperiodic f (-c) := by simpa only [sub_eq_add_neg, Antiperiodic] using h.sub_eq
                              /-
                                🎉 no goals
                              -/


theorem Antiperiodic.neg_eq [AddGroup α] [InvolutiveNeg β] (h : Antiperiodic f c) :
    f (-c) = -f 0 := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    ⊢ Eq (f (Neg.neg c)) (Neg.neg (f 0))
  -/
  simpa only [zero_add] using h.neg 0
  /-
    🎉 no goals
  -/


theorem Antiperiodic.nat_mul_eq_of_eq_zero [Semiring α] [NegZeroClass β] (h : Antiperiodic f c)
    (hi : f 0 = 0) : ∀ n : ℕ, f (n * c) = 0
            /-
              α : Type u_1
              β : Type u_2
              f : α → β
              c : α
              inst✝¹ : Semiring α
              inst✝ : NegZeroClass β
              h : Function.Antiperiodic f c
              hi : Eq (f 0) 0
              ⊢ Eq (f (HMul.hMul (↑0) c)) 0
            -/
  | 0 => by rwa [Nat.cast_zero, zero_mul]
            /-
              🎉 no goals
            -/
                /-
                  α : Type u_1
                  β : Type u_2
                  f : α → β
                  c : α
                  inst✝¹ : Semiring α
                  inst✝ : NegZeroClass β
                  h : Function.Antiperiodic f c
                  hi : Eq (f 0) 0
                  n : Nat
                  ⊢ Eq (f (HMul.hMul (↑(HAdd.hAdd n 1)) c)) 0
                -/
  | n + 1 => by simp [add_mul, h _, Antiperiodic.nat_mul_eq_of_eq_zero h hi n]
                /-
                  🎉 no goals
                -/


theorem Antiperiodic.int_mul_eq_of_eq_zero [Ring α] [SubtractionMonoid β] (h : Antiperiodic f c)
    (hi : f 0 = 0) : ∀ n : ℤ, f (n * c) = 0
                  /-
                    α : Type u_1
                    β : Type u_2
                    f : α → β
                    c : α
                    inst✝¹ : Ring α
                    inst✝ : SubtractionMonoid β
                    h : Function.Antiperiodic f c
                    hi : Eq (f 0) 0
                    n : Nat
                    ⊢ Eq (f (HMul.hMul (↑↑n) c)) 0
                  -/
  | (n : ℕ) => by rw [Int.cast_natCast, h.nat_mul_eq_of_eq_zero hi n]
                  /-
                    🎉 no goals
                  -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       f : α → β
                       c : α
                       inst✝¹ : Ring α
                       inst✝ : SubtractionMonoid β
                       h : Function.Antiperiodic f c
                       hi : Eq (f 0) 0
                       n : Nat
                       ⊢ Eq (f (HMul.hMul (↑(Int.negSucc n)) c)) 0
                     -/
  | .negSucc n => by rw [Int.cast_negSucc, neg_mul, ← mul_neg, h.neg.nat_mul_eq_of_eq_zero hi]
                     /-
                       🎉 no goals
                     -/


theorem Antiperiodic.add_zsmul_eq [AddGroup α] [AddGroup β] (h : Antiperiodic f c) (n : ℤ) :
    f (x + n • c) = (n.negOnePow : ℤ) • f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddGroup α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul n c))) (HSMul.hSMul (↑n.negOnePow) (f x))
  -/
  rcases Int.even_or_odd' n with ⟨k, rfl | rfl⟩
    /-
      case intro.inl
      α : Type u_1
      β : Type u_2
      f : α → β
      c x : α
      inst✝¹ : AddGroup α
      inst✝ : AddGroup β
      h : Function.Antiperiodic f c
      k : Int
      ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul (HMul.hMul 2 k) c))) (HSMul.hSMul (↑(HMul.hM …
    -/
  · rw [h.even_zsmul_periodic, Int.negOnePow_two_mul, Units.val_one, one_zsmul]
    /-
      🎉 no goals
    -/
  · rw [h.odd_zsmul_antiperiodic, Int.negOnePow_two_mul_add_one, Units.val_neg,
      Units.val_one, neg_zsmul, one_zsmul]


theorem Antiperiodic.sub_zsmul_eq [AddGroup α] [AddGroup β] (h : Antiperiodic f c) (n : ℤ) :
    f (x - n • c) = (n.negOnePow : ℤ) • f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddGroup α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Eq (f (HSub.hSub x (HSMul.hSMul n c))) (HSMul.hSMul (↑n.negOnePow) (f x))
  -/
  simpa only [sub_eq_add_neg, neg_zsmul, Int.negOnePow_neg] using h.add_zsmul_eq (-n)
  /-
    🎉 no goals
  -/


theorem Antiperiodic.zsmul_sub_eq [AddCommGroup α] [AddGroup β] (h : Antiperiodic f c) (n : ℤ) :
    f (n • c - x) = (n.negOnePow : ℤ) • f (-x) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddCommGroup α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Eq (f (HSub.hSub (HSMul.hSMul n c) x)) (HSMul.hSMul (↑n.negOnePow) (f (Neg.n …
  -/
  rw [sub_eq_add_neg, add_comm]
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddCommGroup α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Eq (f (HAdd.hAdd (Neg.neg x) (HSMul.hSMul n c))) (HSMul.hSMul (↑n.negOnePow) …
  -/
  exact h.add_zsmul_eq n
  /-
    🎉 no goals
  -/


theorem Antiperiodic.add_int_mul_eq [Ring α] [Ring β] (h : Antiperiodic f c) (n : ℤ) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    f : α → β
                                                    c x : α
                                                    inst✝¹ : Ring α
                                                    inst✝ : Ring β
                                                    h : Function.Antiperiodic f c
                                                    n : Int
                                                    ⊢ Eq (f (HAdd.hAdd x (HMul.hMul (↑n) c))) (HMul.hMul (↑↑n.negOnePow) (f x))
                                                  -/
    f (x + n * c) = (n.negOnePow : ℤ) * f x := by simpa only [zsmul_eq_mul] using h.add_zsmul_eq n
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem Antiperiodic.sub_int_mul_eq [Ring α] [Ring β] (h : Antiperiodic f c) (n : ℤ) :
                                                  /-
                                                    α : Type u_1
                                                    β : Type u_2
                                                    f : α → β
                                                    c x : α
                                                    inst✝¹ : Ring α
                                                    inst✝ : Ring β
                                                    h : Function.Antiperiodic f c
                                                    n : Int
                                                    ⊢ Eq (f (HSub.hSub x (HMul.hMul (↑n) c))) (HMul.hMul (↑↑n.negOnePow) (f x))
                                                  -/
    f (x - n * c) = (n.negOnePow : ℤ) * f x := by simpa only [zsmul_eq_mul] using h.sub_zsmul_eq n
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem Antiperiodic.int_mul_sub_eq [Ring α] [Ring β] (h : Antiperiodic f c) (n : ℤ) :
    f (n * c - x) = (n.negOnePow : ℤ) * f (-x) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : Ring α
    inst✝ : Ring β
    h : Function.Antiperiodic f c
    n : Int
    ⊢ Eq (f (HSub.hSub (HMul.hMul (↑n) c) x)) (HMul.hMul (↑↑n.negOnePow) (f (Neg.n …
  -/
  simpa only [zsmul_eq_mul] using h.zsmul_sub_eq n
  /-
    🎉 no goals
  -/


theorem Antiperiodic.add_nsmul_eq [AddMonoid α] [AddGroup β] (h : Antiperiodic f c) (n : ℕ) :
    f (x + n • c) = (-1) ^ n • f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddMonoid α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Nat
    ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul n c))) (HSMul.hSMul (HPow.hPow (-1) n) (f x))
  -/
  rcases Nat.even_or_odd' n with ⟨k, rfl | rfl⟩
    /-
      case intro.inl
      α : Type u_1
      β : Type u_2
      f : α → β
      c x : α
      inst✝¹ : AddMonoid α
      inst✝ : AddGroup β
      h : Function.Antiperiodic f c
      k : Nat
      ⊢ Eq (f (HAdd.hAdd x (HSMul.hSMul (HMul.hMul 2 k) c))) (HSMul.hSMul (HPow.hPow …
    -/
  · rw [h.even_nsmul_periodic, pow_mul, (by norm_num : (-1) ^ 2 = 1), one_pow, one_zsmul]
    /-
      🎉 no goals
    -/
  · rw [h.odd_nsmul_antiperiodic, pow_add, pow_mul, (by norm_num : (-1) ^ 2 = 1), one_pow,
      pow_one, one_mul, neg_zsmul, one_zsmul]


theorem Antiperiodic.sub_nsmul_eq [AddGroup α] [AddGroup β] (h : Antiperiodic f c) (n : ℕ) :
    f (x - n • c) = (-1) ^ n • f x := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddGroup α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Nat
    ⊢ Eq (f (HSub.hSub x (HSMul.hSMul n c))) (HSMul.hSMul (HPow.hPow (-1) n) (f x))
  -/
  simpa only [Int.reduceNeg, natCast_zsmul] using h.sub_zsmul_eq n
  /-
    🎉 no goals
  -/


theorem Antiperiodic.nsmul_sub_eq [AddCommGroup α] [AddGroup β] (h : Antiperiodic f c) (n : ℕ) :
    f (n • c - x) = (-1) ^ n • f (-x) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c x : α
    inst✝¹ : AddCommGroup α
    inst✝ : AddGroup β
    h : Function.Antiperiodic f c
    n : Nat
    ⊢ Eq (f (HSub.hSub (HSMul.hSMul n c) x)) (HSMul.hSMul (HPow.hPow (-1) n) (f (N …
  -/
  simpa only [Int.reduceNeg, natCast_zsmul] using h.zsmul_sub_eq n
  /-
    🎉 no goals
  -/


theorem Antiperiodic.add_nat_mul_eq [Semiring α] [Ring β] (h : Antiperiodic f c) (n : ℕ) :
    f (x + n * c) = (-1) ^ n * f x := by
  simpa only [nsmul_eq_mul, zsmul_eq_mul, Int.cast_pow, Int.cast_neg,
    Int.cast_one] using h.add_nsmul_eq n


theorem Antiperiodic.sub_nat_mul_eq [Ring α] [Ring β] (h : Antiperiodic f c) (n : ℕ) :
    f (x - n * c) = (-1) ^ n * f x := by
  simpa only [nsmul_eq_mul, zsmul_eq_mul, Int.cast_pow, Int.cast_neg,
    Int.cast_one] using h.sub_nsmul_eq n


theorem Antiperiodic.nat_mul_sub_eq [Ring α] [Ring β] (h : Antiperiodic f c) (n : ℕ) :
    f (n * c - x) = (-1) ^ n * f (-x) := by
  simpa only [nsmul_eq_mul, zsmul_eq_mul, Int.cast_pow, Int.cast_neg,
    Int.cast_one] using h.nsmul_sub_eq n


theorem Antiperiodic.const_add [AddSemigroup α] [Neg β] (h : Antiperiodic f c) (a : α) :
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         f : α → β
                                                         c : α
                                                         inst✝¹ : AddSemigroup α
                                                         inst✝ : Neg β
                                                         h : Function.Antiperiodic f c
                                                         a x : α
                                                         ⊢ Eq ((fun x => f (HAdd.hAdd a x)) (HAdd.hAdd x c)) (Neg.neg ((fun x => f (HAd …
                                                       -/
    Antiperiodic (fun x => f (a + x)) c := fun x => by simpa [add_assoc] using h (a + x)
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Antiperiodic.add_const [AddCommSemigroup α] [Neg β] (h : Antiperiodic f c) (a : α) :
    Antiperiodic (fun x => f (x + a)) c := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddCommSemigroup α
    inst✝ : Neg β
    h : Function.Antiperiodic f c
    a x : α
    ⊢ Eq ((fun x => f (HAdd.hAdd x a)) (HAdd.hAdd x c)) (Neg.neg ((fun x => f (HAd …
  -/
  simpa only [add_right_comm] using h (x + a)
  /-
    🎉 no goals
  -/


theorem Antiperiodic.const_sub [AddCommGroup α] [InvolutiveNeg β] (h : Antiperiodic f c) (a : α) :
    Antiperiodic (fun x => f (a - x)) c := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddCommGroup α
    inst✝ : InvolutiveNeg β
    h : Function.Antiperiodic f c
    a x : α
    ⊢ Eq ((fun x => f (HSub.hSub a x)) (HAdd.hAdd x c)) (Neg.neg ((fun x => f (HSu …
  -/
  simp only [← sub_sub, h.sub_eq]
  /-
    🎉 no goals
  -/


theorem Antiperiodic.sub_const [AddCommGroup α] [Neg β] (h : Antiperiodic f c) (a : α) :
    Antiperiodic (fun x => f (x - a)) c := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : AddCommGroup α
    inst✝ : Neg β
    h : Function.Antiperiodic f c
    a : α
    ⊢ Function.Antiperiodic (fun x => f (HSub.hSub x a)) c
  -/
  simpa only [sub_eq_add_neg] using h.add_const (-a)
  /-
    🎉 no goals
  -/


theorem Antiperiodic.smul [Add α] [Monoid γ] [AddGroup β] [DistribMulAction γ β]
                                                                  /-
                                                                    α : Type u_1
                                                                    β : Type u_2
                                                                    γ : Type u_3
                                                                    f : α → β
                                                                    c : α
                                                                    inst✝³ : Add α
                                                                    inst✝² : Monoid γ
                                                                    inst✝¹ : AddGroup β
                                                                    inst✝ : DistribMulAction γ β
                                                                    h : Function.Antiperiodic f c
                                                                    a : γ
                                                                    ⊢ Function.Antiperiodic (HSMul.hSMul a f) c
                                                                  -/
    (h : Antiperiodic f c) (a : γ) : Antiperiodic (a • f) c := by simp_all
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem Antiperiodic.const_smul [AddMonoid α] [Neg β] [Group γ] [DistribMulAction γ α]
    (h : Antiperiodic f c) (a : γ) : Antiperiodic (fun x => f (a • x)) (a⁻¹ • c) := fun x => by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝³ : AddMonoid α
    inst✝² : Neg β
    inst✝¹ : Group γ
    inst✝ : DistribMulAction γ α
    h : Function.Antiperiodic f c
    a : γ
    x : α
    ⊢ Eq ((fun x => f (HSMul.hSMul a x)) (HAdd.hAdd x (HSMul.hSMul (Inv.inv a) c)) …
  -/
  simpa only [smul_add, smul_inv_smul] using h (a • x)
  /-
    🎉 no goals
  -/


theorem Antiperiodic.const_smul₀ [AddCommMonoid α] [Neg β] [DivisionSemiring γ] [Module γ α]
    (h : Antiperiodic f c) {a : γ} (ha : a ≠ 0) : Antiperiodic (fun x => f (a • x)) (a⁻¹ • c) :=
              /-
                α : Type u_1
                β : Type u_2
                γ : Type u_3
                f : α → β
                c : α
                inst✝³ : AddCommMonoid α
                inst✝² : Neg β
                inst✝¹ : DivisionSemiring γ
                inst✝ : Module γ α
                h : Function.Antiperiodic f c
                a : γ
                ha : Ne a 0
                x : α
                ⊢ Eq ((fun x => f (HSMul.hSMul a x)) (HAdd.hAdd x (HSMul.hSMul (Inv.inv a) c)) …
              -/
  fun x => by simpa only [smul_add, smul_inv_smul₀ ha] using h (a • x)
              /-
                🎉 no goals
              -/


theorem Antiperiodic.const_mul [DivisionSemiring α] [Neg β] (h : Antiperiodic f c) {a : α}
    (ha : a ≠ 0) : Antiperiodic (fun x => f (a * x)) (a⁻¹ * c) :=
  h.const_smul₀ ha


theorem Antiperiodic.const_inv_smul [AddMonoid α] [Neg β] [Group γ] [DistribMulAction γ α]
    (h : Antiperiodic f c) (a : γ) : Antiperiodic (fun x => f (a⁻¹ • x)) (a • c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝³ : AddMonoid α
    inst✝² : Neg β
    inst✝¹ : Group γ
    inst✝ : DistribMulAction γ α
    h : Function.Antiperiodic f c
    a : γ
    ⊢ Function.Antiperiodic (fun x => f (HSMul.hSMul (Inv.inv a) x)) (HSMul.hSMul  …
  -/
  simpa only [inv_inv] using h.const_smul a⁻¹
  /-
    🎉 no goals
  -/


theorem Antiperiodic.const_inv_smul₀ [AddCommMonoid α] [Neg β] [DivisionSemiring γ] [Module γ α]
    (h : Antiperiodic f c) {a : γ} (ha : a ≠ 0) : Antiperiodic (fun x => f (a⁻¹ • x)) (a • c) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    f : α → β
    c : α
    inst✝³ : AddCommMonoid α
    inst✝² : Neg β
    inst✝¹ : DivisionSemiring γ
    inst✝ : Module γ α
    h : Function.Antiperiodic f c
    a : γ
    ha : Ne a 0
    ⊢ Function.Antiperiodic (fun x => f (HSMul.hSMul (Inv.inv a) x)) (HSMul.hSMul  …
  -/
  simpa only [inv_inv] using h.const_smul₀ (inv_ne_zero ha)
  /-
    🎉 no goals
  -/


theorem Antiperiodic.const_inv_mul [DivisionSemiring α] [Neg β] (h : Antiperiodic f c) {a : α}
    (ha : a ≠ 0) : Antiperiodic (fun x => f (a⁻¹ * x)) (a * c) :=
  h.const_inv_smul₀ ha


theorem Antiperiodic.mul_const [DivisionSemiring α] [Neg β] (h : Antiperiodic f c) {a : α}
    (ha : a ≠ 0) : Antiperiodic (fun x => f (x * a)) (c * a⁻¹) :=
  h.const_smul₀ <| (MulOpposite.op_ne_zero_iff a).mpr ha


theorem Antiperiodic.mul_const' [DivisionSemiring α] [Neg β] (h : Antiperiodic f c) {a : α}
    (ha : a ≠ 0) : Antiperiodic (fun x => f (x * a)) (c / a) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : DivisionSemiring α
    inst✝ : Neg β
    h : Function.Antiperiodic f c
    a : α
    ha : Ne a 0
    ⊢ Function.Antiperiodic (fun x => f (HMul.hMul x a)) (HDiv.hDiv c a)
  -/
  simpa only [div_eq_mul_inv] using h.mul_const ha
  /-
    🎉 no goals
  -/


theorem Antiperiodic.mul_const_inv [DivisionSemiring α] [Neg β] (h : Antiperiodic f c) {a : α}
    (ha : a ≠ 0) : Antiperiodic (fun x => f (x * a⁻¹)) (c * a) :=
  h.const_inv_smul₀ <| (MulOpposite.op_ne_zero_iff a).mpr ha


theorem Antiperiodic.div_inv [DivisionSemiring α] [Neg β] (h : Antiperiodic f c) {a : α}
    (ha : a ≠ 0) : Antiperiodic (fun x => f (x / a)) (c * a) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c : α
    inst✝¹ : DivisionSemiring α
    inst✝ : Neg β
    h : Function.Antiperiodic f c
    a : α
    ha : Ne a 0
    ⊢ Function.Antiperiodic (fun x => f (HDiv.hDiv x a)) (HMul.hMul c a)
  -/
  simpa only [div_eq_mul_inv] using h.mul_const_inv ha
  /-
    🎉 no goals
  -/


theorem Antiperiodic.add [AddGroup α] [InvolutiveNeg β] (h1 : Antiperiodic f c₁)
                                                          /-
                                                            α : Type u_1
                                                            β : Type u_2
                                                            f : α → β
                                                            c₁ c₂ : α
                                                            inst✝¹ : AddGroup α
                                                            inst✝ : InvolutiveNeg β
                                                            h1 : Function.Antiperiodic f c₁
                                                            h2 : Function.Antiperiodic f c₂
                                                            ⊢ Function.Periodic f (HAdd.hAdd c₁ c₂)
                                                          -/
    (h2 : Antiperiodic f c₂) : Periodic f (c₁ + c₂) := by simp_all [← add_assoc]
                                                          /-
                                                            🎉 no goals
                                                          -/


theorem Antiperiodic.sub [AddGroup α] [InvolutiveNeg β] (h1 : Antiperiodic f c₁)
    (h2 : Antiperiodic f c₂) : Periodic f (c₁ - c₂) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c₁ c₂ : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h1 : Function.Antiperiodic f c₁
    h2 : Function.Antiperiodic f c₂
    ⊢ Function.Periodic f (HSub.hSub c₁ c₂)
  -/
  simpa only [sub_eq_add_neg] using h1.add h2.neg
  /-
    🎉 no goals
  -/


theorem Periodic.add_antiperiod [AddGroup α] [Neg β] (h1 : Periodic f c₁) (h2 : Antiperiodic f c₂) :
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     f : α → β
                                     c₁ c₂ : α
                                     inst✝¹ : AddGroup α
                                     inst✝ : Neg β
                                     h1 : Function.Periodic f c₁
                                     h2 : Function.Antiperiodic f c₂
                                     ⊢ Function.Antiperiodic f (HAdd.hAdd c₁ c₂)
                                   -/
    Antiperiodic f (c₁ + c₂) := by simp_all [← add_assoc]
                                   /-
                                     🎉 no goals
                                   -/


theorem Periodic.sub_antiperiod [AddGroup α] [InvolutiveNeg β] (h1 : Periodic f c₁)
    (h2 : Antiperiodic f c₂) : Antiperiodic f (c₁ - c₂) := by
  /-
    α : Type u_1
    β : Type u_2
    f : α → β
    c₁ c₂ : α
    inst✝¹ : AddGroup α
    inst✝ : InvolutiveNeg β
    h1 : Function.Periodic f c₁
    h2 : Function.Antiperiodic f c₂
    ⊢ Function.Antiperiodic f (HSub.hSub c₁ c₂)
  -/
  simpa only [sub_eq_add_neg] using h1.add_antiperiod h2.neg
  /-
    🎉 no goals
  -/


theorem Periodic.add_antiperiod_eq [AddGroup α] [Neg β] (h1 : Periodic f c₁)
    (h2 : Antiperiodic f c₂) : f (c₁ + c₂) = -f 0 :=
  (h1.add_antiperiod h2).eq


theorem Periodic.sub_antiperiod_eq [AddGroup α] [InvolutiveNeg β] (h1 : Periodic f c₁)
    (h2 : Antiperiodic f c₂) : f (c₁ - c₂) = -f 0 :=
  (h1.sub_antiperiod h2).eq


theorem Antiperiodic.mul [Add α] [Mul β] [HasDistribNeg β] (hf : Antiperiodic f c)
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         f g : α → β
                                                         c : α
                                                         inst✝² : Add α
                                                         inst✝¹ : Mul β
                                                         inst✝ : HasDistribNeg β
                                                         hf : Function.Antiperiodic f c
                                                         hg : Function.Antiperiodic g c
                                                         ⊢ Function.Periodic (HMul.hMul f g) c
                                                       -/
    (hg : Antiperiodic g c) : Periodic (f * g) c := by simp_all
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Antiperiodic.div [Add α] [DivisionMonoid β] [HasDistribNeg β] (hf : Antiperiodic f c)
                                                       /-
                                                         α : Type u_1
                                                         β : Type u_2
                                                         f g : α → β
                                                         c : α
                                                         inst✝² : Add α
                                                         inst✝¹ : DivisionMonoid β
                                                         inst✝ : HasDistribNeg β
                                                         hf : Function.Antiperiodic f c
                                                         hg : Function.Antiperiodic g c
                                                         ⊢ Function.Periodic (HDiv.hDiv f g) c
                                                       -/
    (hg : Antiperiodic g c) : Periodic (f / g) c := by simp_all [neg_div_neg_eq]
                                                       /-
                                                         🎉 no goals
                                                       -/


theorem Int.fract_periodic (α) [LinearOrderedRing α] [FloorRing α] :
    Function.Periodic Int.fract (1 : α) := fun a => mod_cast Int.fract_add_int a 1

