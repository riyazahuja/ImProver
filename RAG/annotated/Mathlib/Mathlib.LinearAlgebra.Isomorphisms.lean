/-- The **first isomorphism law for modules**. The quotient of `M` by the kernel of `f` is linearly
equivalent to the range of `f`. -/
noncomputable def quotKerEquivRange : (M ⧸ LinearMap.ker f) ≃ₗ[R] LinearMap.range f :=
  (LinearEquiv.ofInjective (f.ker.liftQ f <| le_rfl) <|
        ker_eq_bot.mp <| Submodule.ker_liftQ_eq_bot _ _ _ (le_refl (LinearMap.ker f))).trans
    (LinearEquiv.ofEq _ _ <| Submodule.range_liftQ _ _ _)


/-- The **first isomorphism theorem for surjective linear maps**. -/
noncomputable def quotKerEquivOfSurjective (f : M →ₗ[R] M₂) (hf : Function.Surjective f) :
    (M ⧸ LinearMap.ker f) ≃ₗ[R] M₂ :=
  f.quotKerEquivRange.trans <| .ofTop (LinearMap.range f) <| range_eq_top.2 hf


@[simp]
theorem quotKerEquivRange_apply_mk (x : M) :
    (f.quotKerEquivRange (Submodule.Quotient.mk x) : M₂) = f x :=
  rfl


@[simp]
theorem quotKerEquivRange_symm_apply_image (x : M) (h : f x ∈ LinearMap.range f) :
    f.quotKerEquivRange.symm ⟨f x, h⟩ = f.ker.mkQ x :=
  f.quotKerEquivRange.symm_apply_apply (f.ker.mkQ x)

-- Porting note: breaking up original definition of quotientInfToSupQuotient to avoid timing out

/-- Linear map from `p` to `p+p'/p'` where `p p'` are submodules of `R` -/
abbrev subToSupQuotient (p p' : Submodule R M) :
    { x // x ∈ p } →ₗ[R] { x // x ∈ p ⊔ p' } ⧸ comap (Submodule.subtype (p ⊔ p')) p' :=
  (comap (p ⊔ p').subtype p').mkQ.comp (Submodule.inclusion le_sup_left)

-- Porting note: breaking up original definition of quotientInfToSupQuotient to avoid timing out

theorem comap_leq_ker_subToSupQuotient (p p' : Submodule R M) :
    comap (Submodule.subtype p) (p ⊓ p') ≤ ker (subToSupQuotient p p') := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ LE.le (Submodule.comap p.subtype (Min.min p p')) (LinearMap.ker (LinearMap.s …
  -/
  rw [LinearMap.ker_comp, Submodule.inclusion, comap_codRestrict, ker_mkQ, map_comap_subtype]
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ LE.le (Submodule.comap p.subtype (Min.min p p')) (Submodule.comap p.subtype  …
  -/
  exact comap_mono (inf_le_inf_right _ le_sup_left)
  /-
    🎉 no goals
  -/


/-- Canonical linear map from the quotient `p/(p ∩ p')` to `(p+p')/p'`, mapping `x + (p ∩ p')`
to `x + p'`, where `p` and `p'` are submodules of an ambient module.
-/
def quotientInfToSupQuotient (p p' : Submodule R M) :
    (↥p) ⧸ (comap p.subtype (p ⊓ p')) →ₗ[R] (↥(p ⊔ p')) ⧸ (comap (p ⊔ p').subtype p') :=
   (comap p.subtype (p ⊓ p')).liftQ (subToSupQuotient p p') (comap_leq_ker_subToSupQuotient p p')

-- Porting note: breaking up original definition of quotientInfEquivSupQuotient to avoid timing out

theorem quotientInfEquivSupQuotient_injective (p p' : Submodule R M) :
    Function.Injective (quotientInfToSupQuotient p p') := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ Function.Injective ⇑(LinearMap.quotientInfToSupQuotient p p')
  -/
  rw [← ker_eq_bot, quotientInfToSupQuotient, ker_liftQ_eq_bot]
  /-
    case h'
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ LE.le (LinearMap.ker (LinearMap.subToSupQuotient p p')) (Submodule.comap p.s …
  -/
  rw [ker_comp, ker_mkQ]
  /-
    case h'
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ LE.le (Submodule.comap (Submodule.inclusion ⋯) (Submodule.comap (Max.max p p …
  -/
  exact fun ⟨x, hx1⟩ hx2 => ⟨hx1, hx2⟩
  /-
    🎉 no goals
  -/

-- Porting note: breaking up original definition of quotientInfEquivSupQuotient to avoid timing out

theorem quotientInfEquivSupQuotient_surjective (p p' : Submodule R M) :
    Function.Surjective (quotientInfToSupQuotient p p') := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ Function.Surjective ⇑(LinearMap.quotientInfToSupQuotient p p')
  -/
  rw [← range_eq_top, quotientInfToSupQuotient, range_liftQ, eq_top_iff']
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    ⊢ ∀ (x : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max p p')  …
  -/
  rintro ⟨x, hx⟩; rcases mem_sup.1 hx with ⟨y, hy, z, hz, rfl⟩
  /-
    case mk.mk.intro.intro.intro.intro
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    x✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max p p') x) ( …
    y : M
    hy : Membership.mem p y
    z : M
    hz : Membership.mem p' z
    hx : Membership.mem (Max.max p p') (HAdd.hAdd y z)
    ⊢ Membership.mem (LinearMap.range (LinearMap.subToSupQuotient p p')) (Quot.mk  …
  -/
  use ⟨y, hy⟩; apply (Submodule.Quotient.eq _).2
  /-
    case h
    R : Type u_1
    M : Type u_2
    inst✝² : Ring R
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    p p' : Submodule R M
    x✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Max.max p p') x) ( …
    y : M
    hy : Membership.mem p y
    z : M
    hz : Membership.mem p' z
    hx : Membership.mem (Max.max p p') (HAdd.hAdd y z)
    ⊢ Membership.mem (Submodule.comap (Max.max p p').subtype p') (HSub.hSub ((Subm …
  -/
  simp only [mem_comap, map_sub, coe_subtype, coe_inclusion, sub_add_cancel_left, neg_mem_iff, hz]
  /-
    🎉 no goals
  -/


/--
Second Isomorphism Law : the canonical map from `p/(p ∩ p')` to `(p+p')/p'` as a linear isomorphism.
-/
noncomputable def quotientInfEquivSupQuotient (p p' : Submodule R M) :
    (p ⧸ comap p.subtype (p ⊓ p')) ≃ₗ[R] _ ⧸ comap (p ⊔ p').subtype p' :=
  LinearEquiv.ofBijective (quotientInfToSupQuotient p p')
    ⟨quotientInfEquivSupQuotient_injective p p', quotientInfEquivSupQuotient_surjective p p'⟩

-- @[simp]
-- Porting note: `simp` affects the type arguments of `DFunLike.coe`, so this theorem can't be
--               a simp theorem anymore, even if it has high priority.

theorem coe_quotientInfToSupQuotient (p p' : Submodule R M) :
    ⇑(quotientInfToSupQuotient p p') = quotientInfEquivSupQuotient p p' :=
  rfl

-- This lemma was always bad, but the linter only noticed after https://github.com/leanprover/lean4/pull/2644

@[simp, nolint simpNF]
theorem quotientInfEquivSupQuotient_apply_mk (p p' : Submodule R M) (x : p) :
    let map := inclusion (le_sup_left : p ≤ p ⊔ p')
    quotientInfEquivSupQuotient p p' (Submodule.Quotient.mk x) =
      @Submodule.Quotient.mk R (p ⊔ p' : Submodule R M) _ _ _ (comap (p ⊔ p').subtype p') (map x) :=
  rfl


theorem quotientInfEquivSupQuotient_symm_apply_left (p p' : Submodule R M) (x : ↥(p ⊔ p'))
    (hx : (x : M) ∈ p) :
    (quotientInfEquivSupQuotient p p').symm (Submodule.Quotient.mk x) =
      Submodule.Quotient.mk ⟨x, hx⟩ :=
  (LinearEquiv.symm_apply_eq _).2 <| by
    -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was `simp`.
    /-
      R : Type u_1
      M : Type u_2
      inst✝² : Ring R
      inst✝¹ : AddCommGroup M
      inst✝ : Module R M
      p p' : Submodule R M
      x : Subtype fun x => Membership.mem (Max.max p p') x
      hx : Membership.mem p ↑x
      ⊢ Eq (Submodule.Quotient.mk x) ((LinearMap.quotientInfEquivSupQuotient p p') ( …
    -/
    rw [quotientInfEquivSupQuotient_apply_mk, inclusion_apply]
    /-
      🎉 no goals
    -/



theorem quotientInfEquivSupQuotient_symm_apply_eq_zero_iff {p p' : Submodule R M} {x : ↥(p ⊔ p')} :
    (quotientInfEquivSupQuotient p p').symm (Submodule.Quotient.mk x) = 0 ↔ (x : M) ∈ p' :=
                                            /-
                                              R : Type u_1
                                              M : Type u_2
                                              inst✝² : Ring R
                                              inst✝¹ : AddCommGroup M
                                              inst✝ : Module R M
                                              p p' : Submodule R M
                                              x : Subtype fun x => Membership.mem (Max.max p p') x
                                              ⊢ Iff (Eq (Submodule.Quotient.mk x) ((LinearMap.quotientInfEquivSupQuotient p  …
                                            -/
  (LinearEquiv.symm_apply_eq _).trans <| by simp
                                            /-
                                              🎉 no goals
                                            -/


theorem quotientInfEquivSupQuotient_symm_apply_right (p p' : Submodule R M) {x : ↥(p ⊔ p')}
    (hx : (x : M) ∈ p') : (quotientInfEquivSupQuotient p p').symm (Submodule.Quotient.mk x)
    = 0 :=
  quotientInfEquivSupQuotient_symm_apply_eq_zero_iff.2 hx


/-- The map from the third isomorphism theorem for modules: `(M / S) / (T / S) → M / T`. -/
def quotientQuotientEquivQuotientAux (h : S ≤ T) : (M ⧸ S) ⧸ T.map S.mkQ →ₗ[R] M ⧸ T :=
  liftQ _ (mapQ S T LinearMap.id h)
    (by
      /-
        R : Type u_1
        M : Type u_2
        M₂ : Type u_3
        M₃ : Type u_4
        inst✝⁶ : Ring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        f : LinearMap (RingHom.id R) M M₂
        S T : Submodule R M
        h✝ h : LE.le S T
        ⊢ LE.le (Submodule.map S.mkQ T) (LinearMap.ker (S.mapQ T LinearMap.id h))
      -/
      rintro _ ⟨x, hx, rfl⟩
      /-
        case intro.intro
        R : Type u_1
        M : Type u_2
        M₂ : Type u_3
        M₃ : Type u_4
        inst✝⁶ : Ring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        f : LinearMap (RingHom.id R) M M₂
        S T : Submodule R M
        h✝ h : LE.le S T
        x : M
        hx : Membership.mem (↑T) x
        ⊢ Membership.mem (LinearMap.ker (S.mapQ T LinearMap.id h)) (S.mkQ x)
      -/
      rw [LinearMap.mem_ker, mkQ_apply, mapQ_apply]
      /-
        case intro.intro
        R : Type u_1
        M : Type u_2
        M₂ : Type u_3
        M₃ : Type u_4
        inst✝⁶ : Ring R
        inst✝⁵ : AddCommGroup M
        inst✝⁴ : AddCommGroup M₂
        inst✝³ : AddCommGroup M₃
        inst✝² : Module R M
        inst✝¹ : Module R M₂
        inst✝ : Module R M₃
        f : LinearMap (RingHom.id R) M M₂
        S T : Submodule R M
        h✝ h : LE.le S T
        x : M
        hx : Membership.mem (↑T) x
        ⊢ Eq (Submodule.Quotient.mk (LinearMap.id x)) 0
      -/
      exact (Quotient.mk_eq_zero _).mpr hx)
      /-
        🎉 no goals
      -/


@[simp]
theorem quotientQuotientEquivQuotientAux_mk (x : M ⧸ S) :
    quotientQuotientEquivQuotientAux S T h (Quotient.mk x) = mapQ S T LinearMap.id h x :=
  liftQ_apply _ _ _


@[simp]
theorem quotientQuotientEquivQuotientAux_mk_mk (x : M) :
                                                                                               /-
                                                                                                 R : Type u_1
                                                                                                 M : Type u_2
                                                                                                 inst✝² : Ring R
                                                                                                 inst✝¹ : AddCommGroup M
                                                                                                 inst✝ : Module R M
                                                                                                 S T : Submodule R M
                                                                                                 h : LE.le S T
                                                                                                 x : M
                                                                                                 ⊢ Eq ((S.quotientQuotientEquivQuotientAux T h) (Submodule.Quotient.mk (Submodu …
                                                                                               -/
    quotientQuotientEquivQuotientAux S T h (Quotient.mk (Quotient.mk x)) = Quotient.mk x := by simp
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/


/-- **Noether's third isomorphism theorem** for modules: `(M / S) / (T / S) ≃ M / T`. -/
def quotientQuotientEquivQuotient : ((M ⧸ S) ⧸ T.map S.mkQ) ≃ₗ[R] M ⧸ T :=
  { quotientQuotientEquivQuotientAux S T h with
    toFun := quotientQuotientEquivQuotientAux S T h
    invFun := mapQ _ _ (mkQ S) (le_comap_map _ _)
    left_inv := fun x => Submodule.Quotient.induction_on _
     x fun x => Submodule.Quotient.induction_on _ x fun x =>
         /-
           R : Type u_1
           M : Type u_2
           M₂ : Type u_3
           M₃ : Type u_4
           inst✝⁶ : Ring R
           inst✝⁵ : AddCommGroup M
           inst✝⁴ : AddCommGroup M₂
           inst✝³ : AddCommGroup M₃
           inst✝² : Module R M
           inst✝¹ : Module R M₂
           inst✝ : Module R M₃
           f : LinearMap (RingHom.id R) M M₂
           S T : Submodule R M
           h : LE.le S T
           x✝¹ : HasQuotient.Quotient (HasQuotient.Quotient M S) (Submodule.map S.mkQ T)
           x✝ : HasQuotient.Quotient M S
           x : M
           ⊢ Eq ((T.mapQ (Submodule.map S.mkQ T) S.mkQ ⋯) ({ toFun := ⇑(S.quotientQuotien …
         -/
      by simp
         /-
           🎉 no goals
         -/
    right_inv := fun x => Submodule.Quotient.induction_on _ x
                  /-
                    R : Type u_1
                    M : Type u_2
                    M₂ : Type u_3
                    M₃ : Type u_4
                    inst✝⁶ : Ring R
                    inst✝⁵ : AddCommGroup M
                    inst✝⁴ : AddCommGroup M₂
                    inst✝³ : AddCommGroup M₃
                    inst✝² : Module R M
                    inst✝¹ : Module R M₂
                    inst✝ : Module R M₃
                    f : LinearMap (RingHom.id R) M M₂
                    S T : Submodule R M
                    h : LE.le S T
                    x✝ : HasQuotient.Quotient M T
                    x : M
                    ⊢ Eq ({ toFun := ⇑(S.quotientQuotientEquivQuotientAux T h), map_add' := ⋯, map …
                  -/
      fun x => by simp }
                  /-
                    🎉 no goals
                  -/


/-- Essentially the same equivalence as in the third isomorphism theorem,
except restated in terms of suprema/addition of submodules instead of `≤`. -/
def quotientQuotientEquivQuotientSup : ((M ⧸ S) ⧸ T.map S.mkQ) ≃ₗ[R] M ⧸ S ⊔ T :=
                        /-
                          R : Type u_1
                          M : Type u_2
                          M₂ : Type u_3
                          M₃ : Type u_4
                          inst✝⁶ : Ring R
                          inst✝⁵ : AddCommGroup M
                          inst✝⁴ : AddCommGroup M₂
                          inst✝³ : AddCommGroup M₃
                          inst✝² : Module R M
                          inst✝¹ : Module R M₂
                          inst✝ : Module R M₃
                          f : LinearMap (RingHom.id R) M M₂
                          S T : Submodule R M
                          h : LE.le S T
                          ⊢ Eq (Submodule.map S.mkQ T) (Submodule.map S.mkQ (Max.max S T))
                        -/
  quotEquivOfEq _ _ (by rw [map_sup, mkQ_map_self, bot_sup_eq]) ≪≫ₗ
                        /-
                          🎉 no goals
                        -/
    quotientQuotientEquivQuotient S (S ⊔ T) le_sup_left


/-- Corollary of the third isomorphism theorem: `[S : T] [M : S] = [M : T]` -/
theorem card_quotient_mul_card_quotient (S T : Submodule R M) (hST : T ≤ S) :
    Nat.card (S.map T.mkQ) * Nat.card (M ⧸ S) = Nat.card (M ⧸ T) := by
  rw [Submodule.card_eq_card_quotient_mul_card (map T.mkQ S),
    Nat.card_congr (quotientQuotientEquivQuotient T S hST).toEquiv]


