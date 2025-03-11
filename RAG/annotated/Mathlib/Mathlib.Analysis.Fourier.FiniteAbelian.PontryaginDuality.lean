/-- Indexing of the complex characters of `ZMod n`. `AddChar.zmod n x` is the character sending `y`
to `e ^ (2 * π * i * x * y / n)`. -/
def zmod (x : ZMod n) : AddChar (ZMod n) Circle :=
  AddChar.compAddMonoidHom ⟨AddCircle.toCircle, AddCircle.toCircle_zero, AddCircle.toCircle_add⟩ <|
    ZMod.toAddCircle.comp <| .mulLeft x


@[simp] lemma zmod_intCast (x y : ℤ) : zmod n x y = exp (2 * π * (x * y / n)) := by
  simp [zmod, ← Int.cast_mul x y, -Int.cast_mul, ZMod.toAddCircle_intCast,
    AddCircle.toCircle_apply_mk]


@[simp] lemma zmod_zero : zmod n 0 = 1 :=
                         /-
                           n : Nat
                           inst✝ : NeZero n
                           ⊢ ∀ (x : ZMod n), Eq ((AddChar.zmod n 0) x) (1 x)
                         -/
  DFunLike.ext _ _ <| by simp [ZMod.intCast_surjective.forall, zmod]
                         /-
                           🎉 no goals
                         -/


@[simp] lemma zmod_add : ∀ x y : ZMod n, zmod n (x + y) = zmod n x * zmod n y := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ ∀ (x y : ZMod n), Eq (AddChar.zmod n (HAdd.hAdd x y)) (HMul.hMul (AddChar.zm …
  -/
  simp [DFunLike.ext_iff, ← Int.cast_add, zmod, add_mul, add_div, map_add_eq_mul]
  /-
    🎉 no goals
  -/


lemma zmod_injective : Injective (zmod n) := by
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ Function.Injective (AddChar.zmod n)
  -/
  simp_rw [Injective, ZMod.intCast_surjective.forall]
  /-
    n : Nat
    inst✝ : NeZero n
    ⊢ ∀ (x x_1 : Int), Eq (AddChar.zmod n ↑x) (AddChar.zmod n ↑x_1) → Eq ↑x ↑x_1
  -/
  rintro x y h
  /-
    n : Nat
    inst✝ : NeZero n
    x y : Int
    h : Eq (AddChar.zmod n ↑x) (AddChar.zmod n ↑y)
    ⊢ Eq ↑x ↑y
  -/
  have hn : (n : ℝ) ≠ 0 := NeZero.ne _
  simpa [pi_ne_zero, exp_inj, hn, CharP.intCast_eq_intCast (ZMod n) n] using
    (zmod_intCast ..).symm.trans <| (DFunLike.congr_fun h ((1 : ℤ) : ZMod n)).trans <|
      zmod_intCast ..


@[simp] lemma zmod_inj {x y : ZMod n} : zmod n x = zmod n y ↔ x = y := zmod_injective.eq_iff


/-- `AddChar.zmod` bundled as an `AddChar`. -/
def zmodHom : AddChar (ZMod n) (AddChar (ZMod n) Circle) where
  toFun := zmod n
                         /-
                           α : Type u_1
                           inst✝¹ : AddCommGroup α
                           n✝ : Nat
                           a b : α
                           n : Nat
                           inst✝ : NeZero n
                           ⊢ Eq (AddChar.zmod n 0) 1
                         -/
  map_zero_eq_one' := by simp
                         /-
                           🎉 no goals
                         -/
                        /-
                          α : Type u_1
                          inst✝¹ : AddCommGroup α
                          n✝ : Nat
                          a b : α
                          n : Nat
                          inst✝ : NeZero n
                          ⊢ ∀ (a b : ZMod n), Eq (AddChar.zmod n (HAdd.hAdd a b)) (HMul.hMul (AddChar.zm …
                        -/
  map_add_eq_mul' := by simp
                        /-
                          🎉 no goals
                        -/


/-- Character on a product of `ZMod`s given by `x ↦ ∏ i, e ^ (2 * π * I * x i * y / n)`. -/
private def mkZModAux {ι : Type} [DecidableEq ι] (n : ι → ℕ) [∀ i, NeZero (n i)]
    (u : ∀ i, ZMod (n i)) : AddChar (⨁ i, ZMod (n i)) Circle :=
  AddChar.directSum fun i ↦ zmod (n i) (u i)


private lemma mkZModAux_injective {ι : Type} [DecidableEq ι] {n : ι → ℕ} [∀ i, NeZero (n i)] :
    Injective (mkZModAux n) :=
                                                  /-
                                                    ι : Type
                                                    inst✝¹ : DecidableEq ι
                                                    n : ι → Nat
                                                    inst✝ : ∀ (i : ι), NeZero (n i)
                                                    f g : (i : ι) → ZMod (n i)
                                                    h : Eq (fun i => AddChar.zmod (n i) (f i)) fun i => AddChar.zmod (n i) (g i)
                                                    ⊢ Eq f g
                                                  -/
  AddChar.directSum_injective.comp fun f g h ↦ by simpa [funext_iff] using h
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- The circle-valued characters of a finite abelian group are the same as its complex-valued
characters. -/
def circleEquivComplex [Finite α] : AddChar α Circle ≃+ AddChar α ℂ where
  toFun ψ := toMonoidHomEquiv.symm <| coeHom.comp ψ.toMonoidHom
  invFun ψ :=
    { toFun := fun a ↦ (⟨ψ a, mem_sphere_zero_iff_norm.2 <| ψ.norm_apply _⟩ : Circle)
                             /-
                               α : Type u_1
                               inst✝² : AddCommGroup α
                               n✝ : Nat
                               a b : α
                               n : Nat
                               inst✝¹ : NeZero n
                               inst✝ : Finite α
                               ψ : AddChar α Complex
                               ⊢ Eq ((fun a => ⟨ψ a, ⋯⟩) 0) 1
                             -/
      map_zero_eq_one' := by simp [Circle]
                             /-
                               🎉 no goals
                             -/
                                      /-
                                        α : Type u_1
                                        inst✝² : AddCommGroup α
                                        n✝ : Nat
                                        a✝ b✝ : α
                                        n : Nat
                                        inst✝¹ : NeZero n
                                        inst✝ : Finite α
                                        ψ : AddChar α Complex
                                        a b : α
                                        ⊢ Eq ((fun a => ⟨ψ a, ⋯⟩) (HAdd.hAdd a b)) (HMul.hMul ((fun a => ⟨ψ a, ⋯⟩) a)  …
                                      -/
      map_add_eq_mul' := fun a b ↦ by ext : 1; simp [map_add_eq_mul] }
                                               /-
                                                 🎉 no goals
                                               -/
                   /-
                     α : Type u_1
                     inst✝² : AddCommGroup α
                     n✝ : Nat
                     a b : α
                     n : Nat
                     inst✝¹ : NeZero n
                     inst✝ : Finite α
                     ψ : AddChar α Circle
                     ⊢ Eq ((fun ψ => { toFun := fun a => ⟨ψ a, ⋯⟩, map_zero_eq_one' := ⋯, map_add_e …
                   -/
  left_inv ψ := by ext : 1; simp
                            /-
                              🎉 no goals
                            -/
                    /-
                      α : Type u_1
                      inst✝² : AddCommGroup α
                      n✝ : Nat
                      a b : α
                      n : Nat
                      inst✝¹ : NeZero n
                      inst✝ : Finite α
                      ψ : AddChar α Complex
                      ⊢ Eq ((fun ψ => AddChar.toMonoidHomEquiv.symm (Circle.coeHom.comp ψ.toMonoidHo …
                    -/
  right_inv ψ := by ext : 1; simp
                             /-
                               🎉 no goals
                             -/
  map_add' ψ χ := rfl


@[simp] lemma card_eq [Fintype α] : card (AddChar α ℂ) = card α := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    inst✝ : Fintype α
    ⊢ Eq (Fintype.card (AddChar α Complex)) (Fintype.card α)
  -/
  obtain ⟨ι, _, n, hn, ⟨e⟩⟩ := AddCommGroup.equiv_directSum_zmod_of_finite' α
  classical
  have hn' i : NeZero (n i) := by have := hn i; exact ⟨by positivity⟩
  let f : α → AddChar α ℂ := fun a ↦ coeHom.compAddChar ((mkZModAux n <| e a).compAddMonoidHom e)
  have hf : Injective f := circleEquivComplex.injective.comp
    ((compAddMonoidHom_injective_left _ e.surjective).comp <| mkZModAux_injective.comp <|
      DFunLike.coe_injective.comp <| e.injective.comp Additive.ofMul.injective)
  exact (card_addChar_le _ _).antisymm (Fintype.card_le_of_injective _ hf)


/-- `ZMod n` is (noncanonically) isomorphic to its group of characters. -/
def zmodAddEquiv : ZMod n ≃+ AddChar (ZMod n) ℂ := by
  refine AddEquiv.ofBijective
    (circleEquivComplex.toAddMonoidHom.comp <| AddChar.toAddMonoidHom zmodHom) ?_
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    n✝ : Nat
    a b : α
    n : Nat
    inst✝ : NeZero n
    ⊢ Function.Bijective ⇑(AddChar.circleEquivComplex.toAddMonoidHom.comp AddChar. …
  -/
  rw [Fintype.bijective_iff_injective_and_card, card_eq]
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    n✝ : Nat
    a b : α
    n : Nat
    inst✝ : NeZero n
    ⊢ And (Function.Injective ⇑(AddChar.circleEquivComplex.toAddMonoidHom.comp Add …
  -/
  exact ⟨circleEquivComplex.injective.comp zmod_injective, rfl⟩
  /-
    🎉 no goals
  -/


@[simp] lemma zmodAddEquiv_apply (x : ZMod n) :
    zmodAddEquiv x = circleEquivComplex (zmod n x) := rfl


/-- Complex-valued characters of a finite abelian group `α` form a basis of `α → ℂ`. -/
def complexBasis : Basis (AddChar α ℂ) ℂ (α → ℂ) :=
  basisOfLinearIndependentOfCardEqFinrank (AddChar.linearIndependent _ _) <| by
    /-
      α : Type u_1
      inst✝² : AddCommGroup α
      n✝ : Nat
      a b : α
      n : Nat
      inst✝¹ : NeZero n
      inst✝ : Finite α
      ⊢ Eq (Fintype.card (AddChar α Complex)) (Module.finrank Complex (α → Complex))
    -/
    cases nonempty_fintype α; rw [card_eq, Module.finrank_fintype_fun_eq_card]
                              /-
                                🎉 no goals
                              -/


@[simp, norm_cast]
lemma coe_complexBasis : ⇑(complexBasis α) = ((⇑) : AddChar α ℂ → α → ℂ) := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    inst✝ : Finite α
    ⊢ Eq (⇑(AddChar.complexBasis α)) DFunLike.coe
  -/
  rw [complexBasis, coe_basisOfLinearIndependentOfCardEqFinrank]
  /-
    🎉 no goals
  -/


@[simp]
                                                                        /-
                                                                          α : Type u_1
                                                                          inst✝¹ : AddCommGroup α
                                                                          inst✝ : Finite α
                                                                          ψ : AddChar α Complex
                                                                          ⊢ Eq ((AddChar.complexBasis α) ψ) ⇑ψ
                                                                        -/
lemma complexBasis_apply (ψ : AddChar α ℂ) : complexBasis α ψ = ψ := by rw [coe_complexBasis]
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


lemma exists_apply_ne_zero : (∃ ψ : AddChar α ℂ, ψ a ≠ 1) ↔ a ≠ 0 := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    a : α
    inst✝ : Finite α
    ⊢ Iff (Exists fun ψ => Ne (ψ a) 1) (Ne a 0)
  -/
  refine ⟨?_, fun ha ↦ ?_⟩
    /-
      case refine_1
      α : Type u_1
      inst✝¹ : AddCommGroup α
      a : α
      inst✝ : Finite α
      ⊢ (Exists fun ψ => Ne (ψ a) 1) → Ne a 0
    -/
  · rintro ⟨ψ, hψ⟩ rfl
    /-
      case refine_1.intro
      α : Type u_1
      inst✝¹ : AddCommGroup α
      inst✝ : Finite α
      ψ : AddChar α Complex
      hψ : Ne (ψ 0) 1
      ⊢ False
    -/
    exact hψ ψ.map_zero_eq_one
    /-
      🎉 no goals
    -/
  classical
  by_contra! h
  let f : α → ℂ := fun b ↦ if a = b then 1 else 0
  have h₀ := congr_fun ((complexBasis α).sum_repr f) 0
  have h₁ := congr_fun ((complexBasis α).sum_repr f) a
  simp only [complexBasis_apply, Fintype.sum_apply, Pi.smul_apply, h, smul_eq_mul, mul_one,
    map_zero_eq_one, if_pos rfl, if_neg ha, f] at h₀ h₁
  exact one_ne_zero (h₁.symm.trans h₀)


lemma forall_apply_eq_zero : (∀ ψ : AddChar α ℂ, ψ a = 1) ↔ a = 0 := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    a : α
    inst✝ : Finite α
    ⊢ Iff (∀ (ψ : AddChar α Complex), Eq (ψ a) 1) (Eq a 0)
  -/
  simpa using exists_apply_ne_zero.not
  /-
    🎉 no goals
  -/


lemma doubleDualEmb_injective : Injective (doubleDualEmb : α → AddChar (AddChar α ℂ) ℂ) :=
  doubleDualEmb.ker_eq_bot_iff.1 <| eq_bot_iff.2 fun a ha ↦
                                      /-
                                        α : Type u_1
                                        inst✝¹ : AddCommGroup α
                                        inst✝ : Finite α
                                        a : α
                                        ha : Membership.mem AddChar.doubleDualEmb.ker a
                                        ψ : AddChar α Complex
                                        ⊢ Eq (ψ a) 1
                                      -/
    forall_apply_eq_zero.1 fun ψ ↦ by simpa using DFunLike.congr_fun ha (Additive.ofMul ψ)
                                      /-
                                        🎉 no goals
                                      -/


lemma doubleDualEmb_bijective : Bijective (doubleDualEmb : α → AddChar (AddChar α ℂ) ℂ) := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    inst✝ : Finite α
    ⊢ Function.Bijective ⇑AddChar.doubleDualEmb
  -/
  cases nonempty_fintype α
  exact (Fintype.bijective_iff_injective_and_card _).2
    ⟨doubleDualEmb_injective, card_eq.symm.trans card_eq.symm⟩


@[simp]
lemma doubleDualEmb_inj : (doubleDualEmb a : AddChar (AddChar α ℂ) ℂ) = doubleDualEmb b ↔ a = b :=
  doubleDualEmb_injective.eq_iff


@[simp] lemma doubleDualEmb_eq_zero : (doubleDualEmb a : AddChar (AddChar α ℂ) ℂ) = 0 ↔ a = 0 := by
  /-
    α : Type u_1
    inst✝¹ : AddCommGroup α
    a : α
    inst✝ : Finite α
    ⊢ Iff (Eq (AddChar.doubleDualEmb a) 0) (Eq a 0)
  -/
  rw [← map_zero doubleDualEmb, doubleDualEmb_inj]
  /-
    🎉 no goals
  -/


lemma doubleDualEmb_ne_zero : (doubleDualEmb a : AddChar (AddChar α ℂ) ℂ) ≠ 0 ↔ a ≠ 0 :=
  doubleDualEmb_eq_zero.not


/-- The double dual isomorphism of a finite abelian group. -/
def doubleDualEquiv : α ≃+ AddChar (AddChar α ℂ) ℂ := .ofBijective _ doubleDualEmb_bijective


@[simp]
lemma coe_doubleDualEquiv : ⇑(doubleDualEquiv : α ≃+ AddChar (AddChar α ℂ) ℂ) = doubleDualEmb := rfl


@[simp] lemma doubleDualEmb_doubleDualEquiv_symm_apply (a : AddChar (AddChar α ℂ) ℂ) :
    doubleDualEmb (doubleDualEquiv.symm a) = a :=
  doubleDualEquiv.apply_symm_apply _


@[simp] lemma doubleDualEquiv_symm_doubleDualEmb_apply (a : AddChar (AddChar α ℂ) ℂ) :
    doubleDualEquiv.symm (doubleDualEmb a) = a := doubleDualEquiv.symm_apply_apply _


lemma sum_apply_eq_ite [Fintype α] [DecidableEq α] (a : α) :
    ∑ ψ : AddChar α ℂ, ψ a = if a = 0 then (Fintype.card α : ℂ) else 0 := by
  /-
    α : Type u_1
    inst✝² : AddCommGroup α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Finset.univ.sum fun ψ => ψ a) (ite (Eq a 0) (↑(Fintype.card α)) 0)
  -/
  simpa using sum_eq_ite (doubleDualEmb a : AddChar (AddChar α ℂ) ℂ)
  /-
    🎉 no goals
  -/


lemma expect_apply_eq_ite [Fintype α] [DecidableEq α] (a : α) :
    𝔼 ψ : AddChar α ℂ, ψ a = if a = 0 then 1 else 0 := by
  /-
    α : Type u_1
    inst✝² : AddCommGroup α
    inst✝¹ : Fintype α
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Finset.univ.expect fun ψ => ψ a) (ite (Eq a 0) 1 0)
  -/
  simpa using expect_eq_ite (doubleDualEmb a : AddChar (AddChar α ℂ) ℂ)
  /-
    🎉 no goals
  -/


lemma sum_apply_eq_zero_iff_ne_zero [Finite α] : ∑ ψ : AddChar α ℂ, ψ a = 0 ↔ a ≠ 0 := by
  classical
  cases nonempty_fintype α
  rw [sum_apply_eq_ite, Ne.ite_eq_right_iff]
  exact Nat.cast_ne_zero.2 Fintype.card_ne_zero


lemma sum_apply_ne_zero_iff_eq_zero [Finite α] : ∑ ψ : AddChar α ℂ, ψ a ≠ 0 ↔ a = 0 :=
  sum_apply_eq_zero_iff_ne_zero.not_left


lemma expect_apply_eq_zero_iff_ne_zero [Finite α] : 𝔼 ψ : AddChar α ℂ, ψ a = 0 ↔ a ≠ 0 := by
  classical
  cases nonempty_fintype α
  rw [expect_apply_eq_ite, one_ne_zero.ite_eq_right_iff]


lemma expect_apply_ne_zero_iff_eq_zero [Finite α] : 𝔼 ψ : AddChar α ℂ, ψ a ≠ 0 ↔ a = 0 :=
  expect_apply_eq_zero_iff_ne_zero.not_left


