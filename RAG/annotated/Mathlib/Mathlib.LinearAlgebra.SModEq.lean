/-- A predicate saying two elements of a module are equivalent modulo a submodule. -/
def SModEq (x y : M) : Prop :=
  (Submodule.Quotient.mk x : M ⧸ U) = Submodule.Quotient.mk y


@[inherit_doc] notation:50 x " ≡ " y " [SMOD " N "]" => SModEq N x y


protected theorem SModEq.def :
    x ≡ y [SMOD U] ↔ (Submodule.Quotient.mk x : M ⧸ U) = Submodule.Quotient.mk y :=
  Iff.rfl


                                                   /-
                                                     R : Type u_1
                                                     inst✝² : Ring R
                                                     M : Type u_3
                                                     inst✝¹ : AddCommGroup M
                                                     inst✝ : Module R M
                                                     U : Submodule R M
                                                     x y : M
                                                     ⊢ Iff (SModEq U x y) (Membership.mem U (HSub.hSub x y))
                                                   -/
theorem sub_mem : x ≡ y [SMOD U] ↔ x - y ∈ U := by rw [SModEq.def, Submodule.Quotient.eq]
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem top : x ≡ y [SMOD (⊤ : Submodule R M)] :=
  (Submodule.Quotient.eq ⊤).2 mem_top


@[simp]
theorem bot : x ≡ y [SMOD (⊥ : Submodule R M)] ↔ x = y := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    x y : M
    ⊢ Iff (SModEq Bot.bot x y) (Eq x y)
  -/
  rw [SModEq.def, Submodule.Quotient.eq, mem_bot, sub_eq_zero]
  /-
    🎉 no goals
  -/


@[mono]
theorem mono (HU : U₁ ≤ U₂) (hxy : x ≡ y [SMOD U₁]) : x ≡ y [SMOD U₂] :=
  (Submodule.Quotient.eq U₂).2 <| HU <| (Submodule.Quotient.eq U₁).1 hxy


@[refl]
protected theorem refl (x : M) : x ≡ x [SMOD U] :=
  @rfl _ _


protected theorem rfl : x ≡ x [SMOD U] :=
  SModEq.refl _


instance : IsRefl _ (SModEq U) :=
  ⟨SModEq.refl⟩


@[symm]
nonrec theorem symm (hxy : x ≡ y [SMOD U]) : y ≡ x [SMOD U] :=
  hxy.symm


@[trans]
nonrec theorem trans (hxy : x ≡ y [SMOD U]) (hyz : y ≡ z [SMOD U]) : x ≡ z [SMOD U] :=
  hxy.trans hyz


instance instTrans : Trans (SModEq U) (SModEq U) (SModEq U) where
  trans := trans


theorem add (hxy₁ : x₁ ≡ y₁ [SMOD U]) (hxy₂ : x₂ ≡ y₂ [SMOD U]) : x₁ + x₂ ≡ y₁ + y₂ [SMOD U] := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x₁ x₂ y₁ y₂ : M
    hxy₁ : SModEq U x₁ y₁
    hxy₂ : SModEq U x₂ y₂
    ⊢ SModEq U (HAdd.hAdd x₁ x₂) (HAdd.hAdd y₁ y₂)
  -/
  rw [SModEq.def] at hxy₁ hxy₂ ⊢
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x₁ x₂ y₁ y₂ : M
    hxy₁ : Eq (Submodule.Quotient.mk x₁) (Submodule.Quotient.mk y₁)
    hxy₂ : Eq (Submodule.Quotient.mk x₂) (Submodule.Quotient.mk y₂)
    ⊢ Eq (Submodule.Quotient.mk (HAdd.hAdd x₁ x₂)) (Submodule.Quotient.mk (HAdd.hA …
  -/
  simp_rw [Quotient.mk_add, hxy₁, hxy₂]
  /-
    🎉 no goals
  -/


theorem smul (hxy : x ≡ y [SMOD U]) (c : R) : c • x ≡ c • y [SMOD U] := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : SModEq U x y
    c : R
    ⊢ SModEq U (HSMul.hSMul c x) (HSMul.hSMul c y)
  -/
  rw [SModEq.def] at hxy ⊢
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : Eq (Submodule.Quotient.mk x) (Submodule.Quotient.mk y)
    c : R
    ⊢ Eq (Submodule.Quotient.mk (HSMul.hSMul c x)) (Submodule.Quotient.mk (HSMul.h …
  -/
  simp_rw [Quotient.mk_smul, hxy]
  /-
    🎉 no goals
  -/


lemma nsmul (hxy : x ≡ y [SMOD U]) (n : ℕ) : n • x ≡ n • y [SMOD U] := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : SModEq U x y
    n : Nat
    ⊢ SModEq U (HSMul.hSMul n x) (HSMul.hSMul n y)
  -/
  rw [SModEq.def] at hxy ⊢
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : Eq (Submodule.Quotient.mk x) (Submodule.Quotient.mk y)
    n : Nat
    ⊢ Eq (Submodule.Quotient.mk (HSMul.hSMul n x)) (Submodule.Quotient.mk (HSMul.h …
  -/
  simp_rw [Quotient.mk_smul, hxy]
  /-
    🎉 no goals
  -/


lemma zsmul (hxy : x ≡ y [SMOD U]) (n : ℤ) : n • x ≡ n • y [SMOD U] := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : SModEq U x y
    n : Int
    ⊢ SModEq U (HSMul.hSMul n x) (HSMul.hSMul n y)
  -/
  rw [SModEq.def] at hxy ⊢
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : Eq (Submodule.Quotient.mk x) (Submodule.Quotient.mk y)
    n : Int
    ⊢ Eq (Submodule.Quotient.mk (HSMul.hSMul n x)) (Submodule.Quotient.mk (HSMul.h …
  -/
  simp_rw [Quotient.mk_smul, hxy]
  /-
    🎉 no goals
  -/


theorem mul {I : Ideal A} {x₁ x₂ y₁ y₂ : A} (hxy₁ : x₁ ≡ y₁ [SMOD I])
    (hxy₂ : x₂ ≡ y₂ [SMOD I]) : x₁ * x₂ ≡ y₁ * y₂ [SMOD I] := by
  /-
    A : Type u_2
    inst✝ : CommRing A
    I : Ideal A
    x₁ x₂ y₁ y₂ : A
    hxy₁ : SModEq I x₁ y₁
    hxy₂ : SModEq I x₂ y₂
    ⊢ SModEq I (HMul.hMul x₁ x₂) (HMul.hMul y₁ y₂)
  -/
  simp only [SModEq.def, Ideal.Quotient.mk_eq_mk, map_mul] at hxy₁ hxy₂ ⊢
  /-
    A : Type u_2
    inst✝ : CommRing A
    I : Ideal A
    x₁ x₂ y₁ y₂ : A
    hxy₁ : Eq ((Ideal.Quotient.mk I) x₁) ((Ideal.Quotient.mk I) y₁)
    hxy₂ : Eq ((Ideal.Quotient.mk I) x₂) ((Ideal.Quotient.mk I) y₂)
    ⊢ Eq (HMul.hMul ((Ideal.Quotient.mk I) x₁) ((Ideal.Quotient.mk I) x₂)) (HMul.h …
  -/
  rw [hxy₁, hxy₂]
  /-
    🎉 no goals
  -/


lemma pow {I : Ideal A} {x y : A} (n : ℕ) (hxy : x ≡ y [SMOD I]) :
    x ^ n ≡ y ^ n [SMOD I] := by
  /-
    A : Type u_2
    inst✝ : CommRing A
    I : Ideal A
    x y : A
    n : Nat
    hxy : SModEq I x y
    ⊢ SModEq I (HPow.hPow x n) (HPow.hPow y n)
  -/
  simp only [SModEq.def, Ideal.Quotient.mk_eq_mk, map_pow] at hxy ⊢
  /-
    A : Type u_2
    inst✝ : CommRing A
    I : Ideal A
    x y : A
    n : Nat
    hxy : Eq ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)
    ⊢ Eq (HPow.hPow ((Ideal.Quotient.mk I) x) n) (HPow.hPow ((Ideal.Quotient.mk I) …
  -/
  rw [hxy]
  /-
    🎉 no goals
  -/


lemma neg (hxy : x ≡ y [SMOD U]) : - x ≡ - y [SMOD U] := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x y : M
    hxy : SModEq U x y
    ⊢ SModEq U (Neg.neg x) (Neg.neg y)
  -/
  simpa only [SModEq.def, Quotient.mk_neg, neg_inj]
  /-
    🎉 no goals
  -/


lemma sub (hxy₁ : x₁ ≡ y₁ [SMOD U]) (hxy₂ : x₂ ≡ y₂ [SMOD U]) : x₁ - x₂ ≡ y₁ - y₂ [SMOD U] := by
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x₁ x₂ y₁ y₂ : M
    hxy₁ : SModEq U x₁ y₁
    hxy₂ : SModEq U x₂ y₂
    ⊢ SModEq U (HSub.hSub x₁ x₂) (HSub.hSub y₁ y₂)
  -/
  rw [SModEq.def] at hxy₁ hxy₂ ⊢
  /-
    R : Type u_1
    inst✝² : Ring R
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    U : Submodule R M
    x₁ x₂ y₁ y₂ : M
    hxy₁ : Eq (Submodule.Quotient.mk x₁) (Submodule.Quotient.mk y₁)
    hxy₂ : Eq (Submodule.Quotient.mk x₂) (Submodule.Quotient.mk y₂)
    ⊢ Eq (Submodule.Quotient.mk (HSub.hSub x₁ x₂)) (Submodule.Quotient.mk (HSub.hS …
  -/
  simp_rw [Quotient.mk_sub, hxy₁, hxy₂]
  /-
    🎉 no goals
  -/


                                            /-
                                              R : Type u_1
                                              inst✝² : Ring R
                                              M : Type u_3
                                              inst✝¹ : AddCommGroup M
                                              inst✝ : Module R M
                                              U : Submodule R M
                                              x : M
                                              ⊢ Iff (SModEq U x 0) (Membership.mem U x)
                                            -/
theorem zero : x ≡ 0 [SMOD U] ↔ x ∈ U := by rw [SModEq.def, Submodule.Quotient.eq, sub_zero]
                                            /-
                                              🎉 no goals
                                            -/


theorem map (hxy : x ≡ y [SMOD U]) (f : M →ₗ[R] N) : f x ≡ f y [SMOD U.map f] :=
  (Submodule.Quotient.eq _).2 <| f.map_sub x y ▸ mem_map_of_mem <| (Submodule.Quotient.eq _).1 hxy


theorem comap {f : M →ₗ[R] N} (hxy : f x ≡ f y [SMOD V]) : x ≡ y [SMOD V.comap f] :=
  (Submodule.Quotient.eq _).2 <|
    show f (x - y) ∈ V from (f.map_sub x y).symm ▸ (Submodule.Quotient.eq _).1 hxy


theorem eval {R : Type*} [CommRing R] {I : Ideal R} {x y : R} (h : x ≡ y [SMOD I]) (f : R[X]) :
    f.eval x ≡ f.eval y [SMOD I] := by
  /-
    R : Type u_5
    inst✝ : CommRing R
    I : Ideal R
    x y : R
    h : SModEq I x y
    f : Polynomial R
    ⊢ SModEq I (Polynomial.eval x f) (Polynomial.eval y f)
  -/
  rw [SModEq.def] at h ⊢
  /-
    R : Type u_5
    inst✝ : CommRing R
    I : Ideal R
    x y : R
    h : Eq (Submodule.Quotient.mk x) (Submodule.Quotient.mk y)
    f : Polynomial R
    ⊢ Eq (Submodule.Quotient.mk (Polynomial.eval x f)) (Submodule.Quotient.mk (Pol …
  -/
  show Ideal.Quotient.mk I (f.eval x) = Ideal.Quotient.mk I (f.eval y)
  /-
    R : Type u_5
    inst✝ : CommRing R
    I : Ideal R
    x y : R
    h : Eq (Submodule.Quotient.mk x) (Submodule.Quotient.mk y)
    f : Polynomial R
    ⊢ Eq ((Ideal.Quotient.mk I) (Polynomial.eval x f)) ((Ideal.Quotient.mk I) (Pol …
  -/
  replace h : Ideal.Quotient.mk I x = Ideal.Quotient.mk I y := h
  /-
    R : Type u_5
    inst✝ : CommRing R
    I : Ideal R
    x y : R
    f : Polynomial R
    h : Eq ((Ideal.Quotient.mk I) x) ((Ideal.Quotient.mk I) y)
    ⊢ Eq ((Ideal.Quotient.mk I) (Polynomial.eval x f)) ((Ideal.Quotient.mk I) (Pol …
  -/
  rw [← Polynomial.eval₂_at_apply, ← Polynomial.eval₂_at_apply, h]
  /-
    🎉 no goals
  -/


