theorem eq_top_of_unit_mem (x y : α) (hx : x ∈ I) (h : y * x = 1) : I = ⊤ :=
  eq_top_iff.2 fun z _ =>
    calc
      z * y * x ∈ I := I.mul_mem_left _ hx
      _ = z * (y * x) := mul_assoc z y x
                  /-
                    α : Type u
                    inst✝ : Semiring α
                    I : Ideal α
                    x y : α
                    hx : Membership.mem I x
                    h : Eq (HMul.hMul y x) 1
                    z : α
                    x✝ : Membership.mem Top.top z
                    ⊢ Eq (HMul.hMul z (HMul.hMul y x)) z
                  -/
      _ = z := by rw [h, mul_one]
                  /-
                    🎉 no goals
                  -/


theorem eq_top_of_isUnit_mem {x} (hx : x ∈ I) (h : IsUnit x) : I = ⊤ :=
  let ⟨y, hy⟩ := h.exists_left_inv
  eq_top_of_unit_mem I x y hx hy


theorem eq_top_iff_one : I = ⊤ ↔ (1 : α) ∈ I :=
      /-
        α : Type u
        inst✝ : Semiring α
        I : Ideal α
        ⊢ Eq I Top.top → Membership.mem I 1
      -/
                  /-
                    🎉 no goals
                  -/
  ⟨by rintro rfl; trivial, fun h => eq_top_of_unit_mem _ _ 1 h (by simp)⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


theorem ne_top_iff_one : I ≠ ⊤ ↔ (1 : α) ∉ I :=
  not_congr I.eq_top_iff_one


theorem mem_sup_left {S T : Ideal R} : ∀ {x : R}, x ∈ S → x ∈ S ⊔ T :=
  @le_sup_left _ _ S T


theorem mem_sup_right {S T : Ideal R} : ∀ {x : R}, x ∈ T → x ∈ S ⊔ T :=
  @le_sup_right _ _ S T


theorem mem_iSup_of_mem {ι : Sort*} {S : ι → Ideal R} (i : ι) : ∀ {x : R}, x ∈ S i → x ∈ iSup S :=
  @le_iSup _ _ _ S _


theorem mem_sSup_of_mem {S : Set (Ideal R)} {s : Ideal R} (hs : s ∈ S) :
    ∀ {x : R}, x ∈ s → x ∈ sSup S :=
  @le_sSup _ _ _ _ hs


theorem mem_sInf {s : Set (Ideal R)} {x : R} : x ∈ sInf s ↔ ∀ ⦃I⦄, I ∈ s → x ∈ I :=
  ⟨fun hx I his => hx I ⟨I, iInf_pos his⟩, fun H _I ⟨_J, hij⟩ => hij ▸ fun _S ⟨hj, hS⟩ => hS ▸ H hj⟩


@[simp 1001] -- Porting note: increased priority to appease `simpNF`
theorem mem_inf {I J : Ideal R} {x : R} : x ∈ I ⊓ J ↔ x ∈ I ∧ x ∈ J :=
  Iff.rfl


@[simp 1001] -- Porting note: increased priority to appease `simpNF`
theorem mem_iInf {ι : Sort*} {I : ι → Ideal R} {x : R} : x ∈ iInf I ↔ ∀ i, x ∈ I i :=
  Submodule.mem_iInf _


@[simp 1001] -- Porting note: increased priority to appease `simpNF`
theorem mem_bot {x : R} : x ∈ (⊥ : Ideal R) ↔ x = 0 :=
  Submodule.mem_bot _


/-- All ideals in a division (semi)ring are trivial. -/
theorem eq_bot_or_top : I = ⊥ ∨ I = ⊤ := by
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    ⊢ Or (Eq I Bot.bot) (Eq I Top.top)
  -/
  rw [or_iff_not_imp_right]
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    ⊢ Not (Eq I Top.top) → Eq I Bot.bot
  -/
  change _ ≠ _ → _
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    ⊢ Ne I Top.top → Eq I Bot.bot
  -/
  rw [Ideal.ne_top_iff_one]
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    ⊢ Not (Membership.mem I 1) → Eq I Bot.bot
  -/
  intro h1
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    h1 : Not (Membership.mem I 1)
    ⊢ Eq I Bot.bot
  -/
  rw [eq_bot_iff]
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    h1 : Not (Membership.mem I 1)
    ⊢ LE.le I Bot.bot
  -/
  intro r hr
  /-
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    h1 : Not (Membership.mem I 1)
    r : K
    hr : Membership.mem I r
    ⊢ Membership.mem Bot.bot r
  -/
  by_cases H : r = 0; · simpa
                        /-
                          🎉 no goals
                        -/
  /-
    case neg
    K : Type u
    inst✝ : DivisionSemiring K
    I : Ideal K
    h1 : Not (Membership.mem I 1)
    r : K
    hr : Membership.mem I r
    H : Not (Eq r 0)
    ⊢ Membership.mem Bot.bot r
  -/
  simpa [H, h1] using I.mul_mem_left r⁻¹ hr
  /-
    🎉 no goals
  -/


