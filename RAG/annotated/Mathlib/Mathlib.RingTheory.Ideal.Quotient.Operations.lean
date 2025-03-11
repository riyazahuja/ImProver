/-- The induced map from the quotient by the kernel to the codomain.

This is an isomorphism if `f` has a right inverse (`quotientKerEquivOfRightInverse`) /
is surjective (`quotientKerEquivOfSurjective`).
-/
def kerLift : R ⧸ ker f →+* S :=
  Ideal.Quotient.lift _ f fun _ => mem_ker.mp


@[simp]
theorem kerLift_mk (r : R) : kerLift f (Ideal.Quotient.mk (ker f) r) = f r :=
  Ideal.Quotient.lift_mk _ _ _


theorem lift_injective_of_ker_le_ideal (I : Ideal R) {f : R →+* S} (H : ∀ a : R, a ∈ I → f a = 0)
    (hI : ker f ≤ I) : Function.Injective (Ideal.Quotient.lift I f H) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hI : LE.le (RingHom.ker f) I
    ⊢ Function.Injective ⇑(Ideal.Quotient.lift I f H)
  -/
  rw [RingHom.injective_iff_ker_eq_bot, RingHom.ker_eq_bot_iff_eq_zero]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hI : LE.le (RingHom.ker f) I
    ⊢ ∀ (x : HasQuotient.Quotient R I), Eq ((Ideal.Quotient.lift I f H) x) 0 → Eq  …
  -/
  intro u hu
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hI : LE.le (RingHom.ker f) I
    u : HasQuotient.Quotient R I
    hu : Eq ((Ideal.Quotient.lift I f H) u) 0
    ⊢ Eq u 0
  -/
  obtain ⟨v, rfl⟩ := Ideal.Quotient.mk_surjective u
  /-
    case intro
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hI : LE.le (RingHom.ker f) I
    v : R
    hu : Eq ((Ideal.Quotient.lift I f H) ((Ideal.Quotient.mk I) v)) 0
    ⊢ Eq ((Ideal.Quotient.mk I) v) 0
  -/
  rw [Ideal.Quotient.lift_mk] at hu
  /-
    case intro
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hI : LE.le (RingHom.ker f) I
    v : R
    hu : Eq (f v) 0
    ⊢ Eq ((Ideal.Quotient.mk I) v) 0
  -/
  rw [Ideal.Quotient.eq_zero_iff_mem]
  /-
    case intro
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    hI : LE.le (RingHom.ker f) I
    v : R
    hu : Eq (f v) 0
    ⊢ Membership.mem I v
  -/
  exact hI (RingHom.mem_ker.mpr hu)
  /-
    🎉 no goals
  -/


/-- The induced map from the quotient by the kernel is injective. -/
theorem kerLift_injective : Function.Injective (kerLift f) :=
                                                      /-
                                                        R : Type u
                                                        S : Type v
                                                        inst✝¹ : CommRing R
                                                        inst✝ : Semiring S
                                                        f : RingHom R S
                                                        a : R
                                                        ⊢ Membership.mem (RingHom.ker f) a → Eq (f a) 0
                                                      -/
  lift_injective_of_ker_le_ideal (ker f) (fun a => by simp only [mem_ker, imp_self]) le_rfl
                                                      /-
                                                        🎉 no goals
                                                      -/



/-- The **first isomorphism theorem for commutative rings**, computable version. -/
def quotientKerEquivOfRightInverse {g : S → R} (hf : Function.RightInverse g f) :
    R ⧸ ker f ≃+* S :=
  { kerLift f with
    toFun := kerLift f
    invFun := Ideal.Quotient.mk (ker f) ∘ g
    left_inv := by
      /-
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : Semiring S
        f : RingHom R S
        g : S → R
        hf : Function.RightInverse g ⇑f
        ⊢ Function.LeftInverse (Function.comp (⇑(Ideal.Quotient.mk (RingHom.ker f))) g …
      -/
      rintro ⟨x⟩
      /-
        case mk
        R : Type u
        S : Type v
        inst✝¹ : CommRing R
        inst✝ : Semiring S
        f : RingHom R S
        g : S → R
        hf : Function.RightInverse g ⇑f
        x✝ : HasQuotient.Quotient R (RingHom.ker f)
        x : R
        ⊢ Eq (Function.comp (⇑(Ideal.Quotient.mk (RingHom.ker f))) g (f.kerLift (Quot. …
      -/
      apply kerLift_injective
      simp only [Submodule.Quotient.quot_mk_eq_mk, Ideal.Quotient.mk_eq_mk, kerLift_mk,
        Function.comp_apply, hf (f x)]
    right_inv := hf }


@[simp]
theorem quotientKerEquivOfRightInverse.apply {g : S → R} (hf : Function.RightInverse g f)
    (x : R ⧸ ker f) : quotientKerEquivOfRightInverse hf x = kerLift f x :=
  rfl


@[simp]
theorem quotientKerEquivOfRightInverse.Symm.apply {g : S → R} (hf : Function.RightInverse g f)
    (x : S) : (quotientKerEquivOfRightInverse hf).symm x = Ideal.Quotient.mk (ker f) (g x) :=
  rfl


variable (R) in
/-- The quotient of a ring by he zero ideal is isomorphic to the ring itself. -/
def _root_.RingEquiv.quotientBot : R ⧸ (⊥ : Ideal R) ≃+* R :=
  (Ideal.quotEquivOfEq (RingHom.ker_coe_equiv <| .refl _).symm).trans <|
    quotientKerEquivOfRightInverse (f := .id R) (g := _root_.id) fun _ ↦ rfl


/-- The **first isomorphism theorem** for commutative rings, surjective case. -/
noncomputable def quotientKerEquivOfSurjective (hf : Function.Surjective f) : R ⧸ (ker f) ≃+* S :=
  quotientKerEquivOfRightInverse (Classical.choose_spec hf.hasRightInverse)


/-- The **first isomorphism theorem** for commutative rings (`RingHom.rangeS` version). -/
noncomputable def quotientKerEquivRangeS (f : R →+* S) : R ⧸ ker f ≃+* f.rangeS :=
  (Ideal.quotEquivOfEq f.ker_rangeSRestrict.symm).trans <|
  quotientKerEquivOfSurjective f.rangeSRestrict_surjective


/-- The **first isomorphism theorem** for commutative rings (`RingHom.range` version). -/
noncomputable def quotientKerEquivRange (f : R →+* S) : R ⧸ ker f ≃+* f.range :=
  (Ideal.quotEquivOfEq f.ker_rangeRestrict.symm).trans <|
    quotientKerEquivOfSurjective f.rangeRestrict_surjective


@[simp]
theorem map_quotient_self (I : Ideal R) : map (Quotient.mk I) I = ⊥ :=
  eq_bot_iff.2 <|
    Ideal.map_le_iff_le_comap.2 fun _ hx =>
      (Submodule.mem_bot (R ⧸ I)).2 <| Ideal.Quotient.eq_zero_iff_mem.2 hx


@[simp]
theorem mk_ker {I : Ideal R} : ker (Quotient.mk I) = I := by
  /-
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    ⊢ Eq (RingHom.ker (Ideal.Quotient.mk I)) I
  -/
  ext
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I : Ideal R
    x✝ : R
    ⊢ Iff (Membership.mem (RingHom.ker (Ideal.Quotient.mk I)) x✝) (Membership.mem  …
  -/
  rw [ker, mem_comap, Submodule.mem_bot, Quotient.eq_zero_iff_mem]
  /-
    🎉 no goals
  -/


theorem map_mk_eq_bot_of_le {I J : Ideal R} (h : I ≤ J) : I.map (Quotient.mk J) = ⊥ := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    ⊢ Eq (Ideal.map (Ideal.Quotient.mk J) I) Bot.bot
  -/
  rw [map_eq_bot_iff_le_ker, mk_ker]
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    ⊢ LE.le I J
  -/
  exact h
  /-
    🎉 no goals
  -/


theorem ker_quotient_lift {I : Ideal R} (f : R →+* S)
    (H : I ≤ ker f) :
    ker (Ideal.Quotient.lift I f H) = f.ker.map (Quotient.mk I) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : LE.le I (RingHom.ker f)
    ⊢ Eq (RingHom.ker (Ideal.Quotient.lift I f H)) (Ideal.map (Ideal.Quotient.mk I …
  -/
  apply Ideal.ext
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : LE.le I (RingHom.ker f)
    ⊢ ∀ (x : HasQuotient.Quotient R I), Iff (Membership.mem (RingHom.ker (Ideal.Qu …
  -/
  intro x
  /-
    case h
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : LE.le I (RingHom.ker f)
    x : HasQuotient.Quotient R I
    ⊢ Iff (Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x) (Membership …
  -/
  constructor
    /-
      case h.mp
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      ⊢ Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x → Membership.mem  …
    -/
  · intro hx
    /-
      case h.mp
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      hx : Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x
      ⊢ Membership.mem (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker f)) x
    -/
    obtain ⟨y, hy⟩ := Quotient.mk_surjective x
    /-
      case h.mp.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      hx : Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x
      y : R
      hy : Eq ((Ideal.Quotient.mk I) y) x
      ⊢ Membership.mem (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker f)) x
    -/
    rw [mem_ker, ← hy, Ideal.Quotient.lift_mk, ← mem_ker] at hx
    /-
      case h.mp.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      y : R
      hx : Membership.mem (RingHom.ker f) y
      hy : Eq ((Ideal.Quotient.mk I) y) x
      ⊢ Membership.mem (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker f)) x
    -/
    rw [← hy, mem_map_iff_of_surjective (Quotient.mk I) Quotient.mk_surjective]
    /-
      case h.mp.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      y : R
      hx : Membership.mem (RingHom.ker f) y
      hy : Eq ((Ideal.Quotient.mk I) y) x
      ⊢ Exists fun x => And (Membership.mem (RingHom.ker f) x) (Eq ((Ideal.Quotient. …
    -/
    exact ⟨y, hx, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      ⊢ Membership.mem (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker f)) x → Members …
    -/
  · intro hx
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      hx : Membership.mem (Ideal.map (Ideal.Quotient.mk I) (RingHom.ker f)) x
      ⊢ Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x
    -/
    rw [mem_map_iff_of_surjective (Quotient.mk I) Quotient.mk_surjective] at hx
    /-
      case h.mpr
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      hx : Exists fun x_1 => And (Membership.mem (RingHom.ker f) x_1) (Eq ((Ideal.Qu …
      ⊢ Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x
    -/
    obtain ⟨y, hy⟩ := hx
    /-
      case h.mpr.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      y : R
      hy : And (Membership.mem (RingHom.ker f) y) (Eq ((Ideal.Quotient.mk I) y) x)
      ⊢ Membership.mem (RingHom.ker (Ideal.Quotient.lift I f H)) x
    -/
    rw [mem_ker, ← hy.right, Ideal.Quotient.lift_mk]
    /-
      case h.mpr.intro
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : LE.le I (RingHom.ker f)
      x : HasQuotient.Quotient R I
      y : R
      hy : And (Membership.mem (RingHom.ker f) y) (Eq ((Ideal.Quotient.mk I) y) x)
      ⊢ Eq (f y) 0
    -/
    exact hy.left
    /-
      🎉 no goals
    -/


lemma injective_lift_iff {I : Ideal R} {f : R →+* S} (H : ∀ (a : R), a ∈ I → f a = 0) :
    Injective (Quotient.lift I f H) ↔ ker f = I := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    ⊢ Iff (Function.Injective ⇑(Ideal.Quotient.lift I f H)) (Eq (RingHom.ker f) I)
  -/
  rw [injective_iff_ker_eq_bot, ker_quotient_lift, map_eq_bot_iff_le_ker, mk_ker]
  /-
    R : Type u
    S : Type v
    inst✝¹ : CommRing R
    inst✝ : Semiring S
    I : Ideal R
    f : RingHom R S
    H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
    ⊢ Iff (LE.le (RingHom.ker f) I) (Eq (RingHom.ker f) I)
  -/
  constructor
    /-
      case mp
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
      ⊢ LE.le (RingHom.ker f) I → Eq (RingHom.ker f) I
    -/
  · exact fun h ↦ le_antisymm h H
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      S : Type v
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      I : Ideal R
      f : RingHom R S
      H : ∀ (a : R), Membership.mem I a → Eq (f a) 0
      ⊢ Eq (RingHom.ker f) I → LE.le (RingHom.ker f) I
    -/
  · rintro rfl; rfl
                /-
                  🎉 no goals
                -/


lemma ker_Pi_Quotient_mk {ι : Type*} (I : ι → Ideal R) :
    ker (Pi.ringHom fun i : ι ↦ Quotient.mk (I i)) = ⨅ i, I i := by
  /-
    R : Type u
    inst✝ : CommRing R
    ι : Type u_1
    I : ι → Ideal R
    ⊢ Eq (RingHom.ker (Pi.ringHom fun i => Ideal.Quotient.mk (I i))) (iInf fun i = …
  -/
  simp [Pi.ker_ringHom, mk_ker]
  /-
    🎉 no goals
  -/


@[simp]
theorem bot_quotient_isMaximal_iff (I : Ideal R) : (⊥ : Ideal (R ⧸ I)).IsMaximal ↔ I.IsMaximal :=
  ⟨fun hI =>
    mk_ker (I := I) ▸
      comap_isMaximal_of_surjective (Quotient.mk I) Quotient.mk_surjective (K := ⊥) (H := hI),
    fun hI => by
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hI : I.IsMaximal
      ⊢ Bot.bot.IsMaximal
    -/
    letI := Quotient.field I
    /-
      R : Type u
      inst✝ : CommRing R
      I : Ideal R
      hI : I.IsMaximal
      this : Field (HasQuotient.Quotient R I) := Ideal.Quotient.field I
      ⊢ Bot.bot.IsMaximal
    -/
    exact bot_isMaximal⟩
    /-
      🎉 no goals
    -/


/-- See also `Ideal.mem_quotient_iff_mem` in case `I ≤ J`. -/
@[simp]
theorem mem_quotient_iff_mem_sup {I J : Ideal R} {x : R} :
    Quotient.mk I x ∈ J.map (Quotient.mk I) ↔ x ∈ J ⊔ I := by
  rw [← mem_comap, comap_map_of_surjective (Quotient.mk I) Quotient.mk_surjective, ←
    ker_eq_comap_bot, mk_ker]


/-- See also `Ideal.mem_quotient_iff_mem_sup` if the assumption `I ≤ J` is not available. -/
theorem mem_quotient_iff_mem {I J : Ideal R} (hIJ : I ≤ J) {x : R} :
    Quotient.mk I x ∈ J.map (Quotient.mk I) ↔ x ∈ J := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    hIJ : LE.le I J
    x : R
    ⊢ Iff (Membership.mem (Ideal.map (Ideal.Quotient.mk I) J) ((Ideal.Quotient.mk  …
  -/
  rw [mem_quotient_iff_mem_sup, sup_eq_left.mpr hIJ]
  /-
    🎉 no goals
  -/


/-- The homomorphism from `R/(⋂ i, f i)` to `∏ i, (R / f i)` featured in the Chinese
  Remainder Theorem. It is bijective if the ideals `f i` are coprime. -/
def quotientInfToPiQuotient (I : ι → Ideal R) : (R ⧸ ⨅ i, I i) →+* ∀ i, R ⧸ I i :=
  Quotient.lift (⨅ i, I i) (Pi.ringHom fun i : ι ↦ Quotient.mk (I i))
        /-
          R : Type u
          S : Type v
          F : Type w
          inst✝¹ : CommRing R
          inst✝ : Semiring S
          ι : Type u_1
          I : ι → Ideal R
          ⊢ ∀ (a : R), Membership.mem (iInf fun i => I i) a → Eq ((Pi.ringHom fun i => I …
        -/
    (by simp [← RingHom.mem_ker, ker_Pi_Quotient_mk])
        /-
          🎉 no goals
        -/


lemma quotientInfToPiQuotient_mk (I : ι → Ideal R) (x : R) :
    quotientInfToPiQuotient I (Quotient.mk _ x) = fun i : ι ↦ Quotient.mk (I i) x :=
rfl


lemma quotientInfToPiQuotient_mk' (I : ι → Ideal R) (x : R) (i : ι) :
    quotientInfToPiQuotient I (Quotient.mk _ x) i = Quotient.mk (I i) x :=
rfl


lemma quotientInfToPiQuotient_inj (I : ι → Ideal R) : Injective (quotientInfToPiQuotient I) := by
  /-
    R : Type u
    inst✝ : CommRing R
    ι : Type u_1
    I : ι → Ideal R
    ⊢ Function.Injective ⇑(Ideal.quotientInfToPiQuotient I)
  -/
  rw [quotientInfToPiQuotient, injective_lift_iff, ker_Pi_Quotient_mk]
  /-
    🎉 no goals
  -/


lemma quotientInfToPiQuotient_surj [Finite ι] {I : ι → Ideal R}
    (hI : Pairwise (IsCoprime on I)) : Surjective (quotientInfToPiQuotient I) := by
  classical
  cases nonempty_fintype ι
  intro g
  choose f hf using fun i ↦ mk_surjective (g i)
  have key : ∀ i, ∃ e : R, mk (I i) e = 1 ∧ ∀ j, j ≠ i → mk (I j) e = 0 := by
    intro i
    have hI' : ∀ j ∈ ({i} : Finset ι)ᶜ, IsCoprime (I i) (I j) := by
      intros j hj
      exact hI (by simpa [ne_comm, isCoprime_iff_add] using hj)
    rcases isCoprime_iff_exists.mp (isCoprime_biInf hI') with ⟨u, hu, e, he, hue⟩
    replace he : ∀ j, j ≠ i → e ∈ I j := by simpa using he
    refine ⟨e, ?_, ?_⟩
    · simp [eq_sub_of_add_eq' hue, map_sub, eq_zero_iff_mem.mpr hu]
    · exact fun j hj ↦ eq_zero_iff_mem.mpr (he j hj)
  choose e he using key
  use mk _ (∑ i, f i*e i)
  ext i
  rw [quotientInfToPiQuotient_mk', map_sum, Fintype.sum_eq_single i]
  · simp [(he i).1, hf]
  · intros j hj
    simp [(he j).2 i hj.symm]


/-- **Chinese Remainder Theorem**. Eisenbud Ex.2.6.
Similar to Atiyah-Macdonald 1.10 and Stacks 00DT -/
noncomputable def quotientInfRingEquivPiQuotient [Finite ι] (f : ι → Ideal R)
    (hf : Pairwise (IsCoprime on f)) : (R ⧸ ⨅ i, f i) ≃+* ∀ i, R ⧸ f i :=
  { Equiv.ofBijective _ ⟨quotientInfToPiQuotient_inj f, quotientInfToPiQuotient_surj hf⟩,
    quotientInfToPiQuotient f with }


/-- Corollary of Chinese Remainder Theorem: if `Iᵢ` are pairwise coprime ideals in a
commutative ring then the canonical map `R → ∏ (R ⧸ Iᵢ)` is surjective. -/
lemma pi_quotient_surjective {R : Type*} [CommRing R] {ι : Type*} [Finite ι] {I : ι → Ideal R}
    (hf : Pairwise fun i j ↦ IsCoprime (I i) (I j)) (x : (i : ι) → R ⧸ I i) :
    ∃ r : R, ∀ i, r = x i := by
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Finite ι
    I : ι → Ideal R
    hf : Pairwise fun i j => IsCoprime (I i) (I j)
    x : (i : ι) → HasQuotient.Quotient R (I i)
    ⊢ Exists fun r => ∀ (i : ι), Eq ((Ideal.Quotient.mk (I i)) r) (x i)
  -/
  obtain ⟨y, rfl⟩ := Ideal.quotientInfToPiQuotient_surj hf x
  /-
    case intro
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Finite ι
    I : ι → Ideal R
    hf : Pairwise fun i j => IsCoprime (I i) (I j)
    y : HasQuotient.Quotient R (iInf fun i => I i)
    ⊢ Exists fun r => ∀ (i : ι), Eq ((Ideal.Quotient.mk (I i)) r) ((Ideal.quotient …
  -/
  obtain ⟨r, rfl⟩ := Ideal.Quotient.mk_surjective y
  /-
    case intro.intro
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Finite ι
    I : ι → Ideal R
    hf : Pairwise fun i j => IsCoprime (I i) (I j)
    r : R
    ⊢ Exists fun r_1 => ∀ (i : ι), Eq ((Ideal.Quotient.mk (I i)) r_1) ((Ideal.quot …
  -/
  exact ⟨r, fun i ↦ rfl⟩
  /-
    🎉 no goals
  -/

-- variant of `IsDedekindDomain.exists_forall_sub_mem_ideal` which doesn't assume Dedekind domain!

/-- Corollary of Chinese Remainder Theorem: if `Iᵢ` are pairwise coprime ideals in a
commutative ring then given elements `xᵢ` you can find `r` with `r - xᵢ ∈ Iᵢ` for all `i`. -/
lemma exists_forall_sub_mem_ideal {R : Type*} [CommRing R] {ι : Type*} [Finite ι]
    {I : ι → Ideal R} (hI : Pairwise fun i j ↦ IsCoprime (I i) (I j)) (x : ι → R) :
    ∃ r : R, ∀ i, r - x i ∈ I i := by
  /-
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Finite ι
    I : ι → Ideal R
    hI : Pairwise fun i j => IsCoprime (I i) (I j)
    x : ι → R
    ⊢ Exists fun r => ∀ (i : ι), Membership.mem (I i) (HSub.hSub r (x i))
  -/
  obtain ⟨y, hy⟩ := Ideal.pi_quotient_surjective hI (fun i ↦ x i)
  /-
    case intro
    R : Type u_2
    inst✝¹ : CommRing R
    ι : Type u_3
    inst✝ : Finite ι
    I : ι → Ideal R
    hI : Pairwise fun i j => IsCoprime (I i) (I j)
    x : ι → R
    y : R
    hy : ∀ (i : ι), Eq ((Ideal.Quotient.mk (I i)) y) ((Ideal.Quotient.mk (I i)) (x …
    ⊢ Exists fun r => ∀ (i : ι), Membership.mem (I i) (HSub.hSub r (x i))
  -/
  exact ⟨y, fun i ↦ (Submodule.Quotient.eq (I i)).mp <| hy i⟩
  /-
    🎉 no goals
  -/


/-- **Chinese remainder theorem**, specialized to two ideals. -/
noncomputable def quotientInfEquivQuotientProd (I J : Ideal R) (coprime : IsCoprime I J) :
    R ⧸ I ⊓ J ≃+* (R ⧸ I) × R ⧸ J :=
  let f : Fin 2 → Ideal R := ![I, J]
  have hf : Pairwise (IsCoprime on f) := by
    /-
      R : Type u
      S : Type v
      F : Type w
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      ι : Type u_1
      I J : Ideal R
      coprime : IsCoprime I J
      f : Fin 2 → Ideal R := Matrix.vecCons I (Matrix.vecCons J Matrix.vecEmpty)
      ⊢ Pairwise (Function.onFun IsCoprime f)
    -/
    intro i j h
    /-
      R : Type u
      S : Type v
      F : Type w
      inst✝¹ : CommRing R
      inst✝ : Semiring S
      ι : Type u_1
      I J : Ideal R
      coprime : IsCoprime I J
      f : Fin 2 → Ideal R := Matrix.vecCons I (Matrix.vecCons J Matrix.vecEmpty)
      i j : Fin 2
      h : Ne i j
      ⊢ Function.onFun IsCoprime f i j
    -/
                                    /-
                                      🎉 no goals
                                    -/
    fin_cases i <;> fin_cases j <;> try contradiction
                                    /-
                                      🎉 no goals
                                    -/
      /-
        case «0».«1»
        R : Type u
        S : Type v
        F : Type w
        inst✝¹ : CommRing R
        inst✝ : Semiring S
        ι : Type u_1
        I J : Ideal R
        coprime : IsCoprime I J
        f : Fin 2 → Ideal R := Matrix.vecCons I (Matrix.vecCons J Matrix.vecEmpty)
        h : Ne ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨1, ⋯⟩)
        ⊢ Function.onFun IsCoprime f ((fun i => i) ⟨0, ⋯⟩) ((fun i => i) ⟨1, ⋯⟩)
      -/
    · assumption
      /-
        🎉 no goals
      -/
      /-
        case «1».«0»
        R : Type u
        S : Type v
        F : Type w
        inst✝¹ : CommRing R
        inst✝ : Semiring S
        ι : Type u_1
        I J : Ideal R
        coprime : IsCoprime I J
        f : Fin 2 → Ideal R := Matrix.vecCons I (Matrix.vecCons J Matrix.vecEmpty)
        h : Ne ((fun i => i) ⟨1, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)
        ⊢ Function.onFun IsCoprime f ((fun i => i) ⟨1, ⋯⟩) ((fun i => i) ⟨0, ⋯⟩)
      -/
    · exact coprime.symm
      /-
        🎉 no goals
      -/
                           /-
                             R : Type u
                             S : Type v
                             F : Type w
                             inst✝¹ : CommRing R
                             inst✝ : Semiring S
                             ι : Type u_1
                             I J : Ideal R
                             coprime : IsCoprime I J
                             f : Fin 2 → Ideal R := Matrix.vecCons I (Matrix.vecCons J Matrix.vecEmpty)
                             hf : Pairwise (Function.onFun IsCoprime f)
                             ⊢ Eq (Min.min I J) (iInf fun i => f i)
                           -/
  (Ideal.quotEquivOfEq (by simp [f, iInf, inf_comm])).trans <|
                           /-
                             🎉 no goals
                           -/
            (Ideal.quotientInfRingEquivPiQuotient f hf).trans <| RingEquiv.piFinTwo fun i => R ⧸ f i


@[simp]
theorem quotientInfEquivQuotientProd_fst (I J : Ideal R) (coprime : IsCoprime I J) (x : R ⧸ I ⊓ J) :
    (quotientInfEquivQuotientProd I J coprime x).fst =
      Ideal.Quotient.factor (I ⊓ J) I inf_le_left x :=
  Quot.inductionOn x fun _ => rfl


@[simp]
theorem quotientInfEquivQuotientProd_snd (I J : Ideal R) (coprime : IsCoprime I J) (x : R ⧸ I ⊓ J) :
    (quotientInfEquivQuotientProd I J coprime x).snd =
      Ideal.Quotient.factor (I ⊓ J) J inf_le_right x :=
  Quot.inductionOn x fun _ => rfl


@[simp]
theorem fst_comp_quotientInfEquivQuotientProd (I J : Ideal R) (coprime : IsCoprime I J) :
    (RingHom.fst _ _).comp
        (quotientInfEquivQuotientProd I J coprime : R ⧸ I ⊓ J →+* (R ⧸ I) × R ⧸ J) =
      Ideal.Quotient.factor (I ⊓ J) I inf_le_left := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    coprime : IsCoprime I J
    ⊢ Eq ((RingHom.fst (HasQuotient.Quotient R I) (HasQuotient.Quotient R J)).comp …
  -/
  apply Quotient.ringHom_ext; ext; rfl
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem snd_comp_quotientInfEquivQuotientProd (I J : Ideal R) (coprime : IsCoprime I J) :
    (RingHom.snd _ _).comp
        (quotientInfEquivQuotientProd I J coprime : R ⧸ I ⊓ J →+* (R ⧸ I) × R ⧸ J) =
      Ideal.Quotient.factor (I ⊓ J) J inf_le_right := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    coprime : IsCoprime I J
    ⊢ Eq ((RingHom.snd (HasQuotient.Quotient R I) (HasQuotient.Quotient R J)).comp …
  -/
  apply Quotient.ringHom_ext; ext; rfl
                                   /-
                                     🎉 no goals
                                   -/


/-- **Chinese remainder theorem**, specialized to two ideals. -/
noncomputable def quotientMulEquivQuotientProd (I J : Ideal R) (coprime : IsCoprime I J) :
    R ⧸ I * J ≃+* (R ⧸ I) × R ⧸ J :=
  Ideal.quotEquivOfEq (inf_eq_mul_of_isCoprime coprime).symm |>.trans <|
    Ideal.quotientInfEquivQuotientProd I J coprime


@[simp]
theorem quotientMulEquivQuotientProd_fst (I J : Ideal R) (coprime : IsCoprime I J) (x : R ⧸ I * J) :
    (quotientMulEquivQuotientProd I J coprime x).fst =
      Ideal.Quotient.factor (I * J) I mul_le_right x :=
  Quot.inductionOn x fun _ => rfl


@[simp]
theorem quotientMulEquivQuotientProd_snd (I J : Ideal R) (coprime : IsCoprime I J) (x : R ⧸ I * J) :
    (quotientMulEquivQuotientProd I J coprime x).snd =
      Ideal.Quotient.factor (I * J) J mul_le_left x :=
  Quot.inductionOn x fun _ => rfl


@[simp]
theorem fst_comp_quotientMulEquivQuotientProd (I J : Ideal R) (coprime : IsCoprime I J) :
    (RingHom.fst _ _).comp
        (quotientMulEquivQuotientProd I J coprime : R ⧸ I * J →+* (R ⧸ I) × R ⧸ J) =
      Ideal.Quotient.factor (I * J) I mul_le_right := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    coprime : IsCoprime I J
    ⊢ Eq ((RingHom.fst (HasQuotient.Quotient R I) (HasQuotient.Quotient R J)).comp …
  -/
  apply Quotient.ringHom_ext; ext; rfl
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem snd_comp_quotientMulEquivQuotientProd (I J : Ideal R) (coprime : IsCoprime I J) :
    (RingHom.snd _ _).comp
        (quotientMulEquivQuotientProd I J coprime : R ⧸ I * J →+* (R ⧸ I) × R ⧸ J) =
      Ideal.Quotient.factor (I * J) J mul_le_left := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    coprime : IsCoprime I J
    ⊢ Eq ((RingHom.snd (HasQuotient.Quotient R I) (HasQuotient.Quotient R J)).comp …
  -/
  apply Quotient.ringHom_ext; ext; rfl
                                   /-
                                     🎉 no goals
                                   -/


/-- The `R₁`-algebra structure on `A/I` for an `R₁`-algebra `A` -/
instance Quotient.algebra {I : Ideal A} : Algebra R₁ (A ⧸ I) :=
  { toRingHom := (Ideal.Quotient.mk I).comp (algebraMap R₁ A)
    smul_def' := fun _ x =>
      Quotient.inductionOn' x fun _ =>
        ((Quotient.mk I).congr_arg <| Algebra.smul_def _ _).trans (RingHom.map_mul _ _ _)
    commutes' := fun _ _ => mul_comm _ _ }

-- Lean can struggle to find this instance later if we don't provide this shortcut
-- Porting note: this can probably now be deleted
-- update: maybe not - removal causes timeouts

instance Quotient.isScalarTower [SMul R₁ R₂] [IsScalarTower R₁ R₂ A] (I : Ideal A) :
                                      /-
                                        R : Type u
                                        S : Type v
                                        F : Type w
                                        inst✝⁸ : CommRing R
                                        inst✝⁷ : Semiring S
                                        R₁ : Type u_1
                                        R₂ : Type u_2
                                        A : Type u_3
                                        B : Type u_4
                                        inst✝⁶ : CommSemiring R₁
                                        inst✝⁵ : CommSemiring R₂
                                        inst✝⁴ : CommRing A
                                        inst✝³ : Algebra R₁ A
                                        inst✝² : Algebra R₂ A
                                        inst✝¹ : SMul R₁ R₂
                                        inst✝ : IsScalarTower R₁ R₂ A
                                        I : Ideal A
                                        ⊢ IsScalarTower R₁ R₂ (HasQuotient.Quotient A I)
                                      -/
    IsScalarTower R₁ R₂ (A ⧸ I) := by infer_instance
                                      /-
                                        🎉 no goals
                                      -/


/-- The canonical morphism `A →ₐ[R₁] A ⧸ I` as morphism of `R₁`-algebras, for `I` an ideal of
`A`, where `A` is an `R₁`-algebra. -/
def Quotient.mkₐ (I : Ideal A) : A →ₐ[R₁] A ⧸ I :=
  ⟨⟨⟨⟨fun a => Submodule.Quotient.mk a, rfl⟩, fun _ _ => rfl⟩, rfl, fun _ _ => rfl⟩, fun _ => rfl⟩


theorem Quotient.algHom_ext {I : Ideal A} {S} [Semiring S] [Algebra R₁ S] ⦃f g : A ⧸ I →ₐ[R₁] S⦄
    (h : f.comp (Quotient.mkₐ R₁ I) = g.comp (Quotient.mkₐ R₁ I)) : f = g :=
  AlgHom.ext fun x => Quotient.inductionOn' x <| AlgHom.congr_fun h


theorem Quotient.alg_map_eq (I : Ideal A) :
    algebraMap R₁ (A ⧸ I) = (algebraMap A (A ⧸ I)).comp (algebraMap R₁ A) :=
  rfl


theorem Quotient.mkₐ_toRingHom (I : Ideal A) :
    (Quotient.mkₐ R₁ I).toRingHom = Ideal.Quotient.mk I :=
  rfl


@[simp]
theorem Quotient.mkₐ_eq_mk (I : Ideal A) : ⇑(Quotient.mkₐ R₁ I) = Quotient.mk I :=
  rfl


@[simp]
theorem Quotient.algebraMap_eq (I : Ideal R) : algebraMap R (R ⧸ I) = Quotient.mk I :=
  rfl


@[simp]
theorem Quotient.mk_comp_algebraMap (I : Ideal A) :
    (Quotient.mk I).comp (algebraMap R₁ A) = algebraMap R₁ (A ⧸ I) :=
  rfl


@[simp]
theorem Quotient.mk_algebraMap (I : Ideal A) (x : R₁) :
    Quotient.mk I (algebraMap R₁ A x) = algebraMap R₁ (A ⧸ I) x :=
  rfl


/-- The canonical morphism `A →ₐ[R₁] I.quotient` is surjective. -/
theorem Quotient.mkₐ_surjective (I : Ideal A) : Function.Surjective (Quotient.mkₐ R₁ I) :=
  Quot.mk_surjective


/-- The kernel of `A →ₐ[R₁] I.quotient` is `I`. -/
@[simp]
theorem Quotient.mkₐ_ker (I : Ideal A) : RingHom.ker (Quotient.mkₐ R₁ I : A →+* A ⧸ I) = I :=
  Ideal.mk_ker


/-- `Ideal.quotient.lift` as an `AlgHom`. -/
def Quotient.liftₐ (I : Ideal A) (f : A →ₐ[R₁] B) (hI : ∀ a : A, a ∈ I → f a = 0) :
    A ⧸ I →ₐ[R₁] B :=
  {-- this is IsScalarTower.algebraMap_apply R₁ A (A ⧸ I) but the file `Algebra.Algebra.Tower`
      -- imports this file.
      Ideal.Quotient.lift
      I (f : A →+* B) hI with
    commutes' := fun r => by
      have : algebraMap R₁ (A ⧸ I) r = algebraMap A (A ⧸ I) (algebraMap R₁ A r) := by
        simp_rw [Algebra.algebraMap_eq_smul_one, smul_assoc, one_smul]
      rw [this, Ideal.Quotient.algebraMap_eq, RingHom.toFun_eq_coe, Ideal.Quotient.lift_mk,
        AlgHom.coe_toRingHom, Algebra.algebraMap_eq_smul_one, Algebra.algebraMap_eq_smul_one,
        map_smul, map_one] }


@[simp]
theorem Quotient.liftₐ_apply (I : Ideal A) (f : A →ₐ[R₁] B) (hI : ∀ a : A, a ∈ I → f a = 0) (x) :
    Ideal.Quotient.liftₐ I f hI x = Ideal.Quotient.lift I (f : A →+* B) hI x :=
  rfl


theorem Quotient.liftₐ_comp (I : Ideal A) (f : A →ₐ[R₁] B) (hI : ∀ a : A, a ∈ I → f a = 0) :
    (Ideal.Quotient.liftₐ I f hI).comp (Ideal.Quotient.mkₐ R₁ I) = f :=
  AlgHom.ext fun _ => (Ideal.Quotient.lift_mk I (f : A →+* B) hI : _)


theorem KerLift.map_smul (f : A →ₐ[R₁] B) (r : R₁) (x : A ⧸ (RingHom.ker f)) :
    f.kerLift (r • x) = r • f.kerLift x := by
  /-
    R₁ : Type u_1
    A : Type u_3
    B : Type u_4
    inst✝⁴ : CommSemiring R₁
    inst✝³ : CommRing A
    inst✝² : Algebra R₁ A
    inst✝¹ : Semiring B
    inst✝ : Algebra R₁ B
    f : AlgHom R₁ A B
    r : R₁
    x : HasQuotient.Quotient A (RingHom.ker f)
    ⊢ Eq (f.kerLift (HSMul.hSMul r x)) (HSMul.hSMul r (f.kerLift x))
  -/
  obtain ⟨a, rfl⟩ := Quotient.mkₐ_surjective R₁ _ x
  /-
    case intro
    R₁ : Type u_1
    A : Type u_3
    B : Type u_4
    inst✝⁴ : CommSemiring R₁
    inst✝³ : CommRing A
    inst✝² : Algebra R₁ A
    inst✝¹ : Semiring B
    inst✝ : Algebra R₁ B
    f : AlgHom R₁ A B
    r : R₁
    a : A
    ⊢ Eq (f.kerLift (HSMul.hSMul r ((Ideal.Quotient.mkₐ R₁ (RingHom.ker f)) a))) ( …
  -/
  exact _root_.map_smul f _ _
  /-
    🎉 no goals
  -/


/-- The induced algebras morphism from the quotient by the kernel to the codomain.

This is an isomorphism if `f` has a right inverse (`quotientKerAlgEquivOfRightInverse`) /
is surjective (`quotientKerAlgEquivOfSurjective`).
-/
def kerLiftAlg (f : A →ₐ[R₁] B) : A ⧸ (RingHom.ker f) →ₐ[R₁] B :=
  AlgHom.mk' (RingHom.kerLift (f : A →+* B)) fun _ _ => KerLift.map_smul f _ _


@[simp]
theorem kerLiftAlg_mk (f : A →ₐ[R₁] B) (a : A) :
    kerLiftAlg f (Quotient.mk (RingHom.ker f) a) = f a := by
  /-
    R₁ : Type u_1
    A : Type u_3
    B : Type u_4
    inst✝⁴ : CommSemiring R₁
    inst✝³ : CommRing A
    inst✝² : Algebra R₁ A
    inst✝¹ : Semiring B
    inst✝ : Algebra R₁ B
    f : AlgHom R₁ A B
    a : A
    ⊢ Eq ((Ideal.kerLiftAlg f) ((Ideal.Quotient.mk (RingHom.ker f)) a)) (f a)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem kerLiftAlg_toRingHom (f : A →ₐ[R₁] B) :
    (kerLiftAlg f : A ⧸ ker f →+* B) = RingHom.kerLift (f : A →+* B) :=
  rfl


/-- The induced algebra morphism from the quotient by the kernel is injective. -/
theorem kerLiftAlg_injective (f : A →ₐ[R₁] B) : Function.Injective (kerLiftAlg f) :=
  RingHom.kerLift_injective (R := A) (S := B) f


/-- The **first isomorphism** theorem for algebras, computable version. -/
@[simps!]
def quotientKerAlgEquivOfRightInverse {f : A →ₐ[R₁] B} {g : B → A}
    (hf : Function.RightInverse g f) : (A ⧸ RingHom.ker f) ≃ₐ[R₁] B :=
  { RingHom.quotientKerEquivOfRightInverse hf,
    kerLiftAlg f with }


@[deprecated (since := "2024-02-27")]
alias quotientKerAlgEquivOfRightInverse.apply := quotientKerAlgEquivOfRightInverse_apply

@[deprecated (since := "2024-02-27")]
alias QuotientKerAlgEquivOfRightInverseSymm.apply := quotientKerAlgEquivOfRightInverse_symm_apply


/-- The **first isomorphism theorem** for algebras. -/
@[simps!]
noncomputable def quotientKerAlgEquivOfSurjective {f : A →ₐ[R₁] B} (hf : Function.Surjective f) :
    (A ⧸ (RingHom.ker f)) ≃ₐ[R₁] B :=
  quotientKerAlgEquivOfRightInverse (Classical.choose_spec hf.hasRightInverse)


/-- The ring hom `R/I →+* S/J` induced by a ring hom `f : R →+* S` with `I ≤ f⁻¹(J)` -/
def quotientMap {I : Ideal R} (J : Ideal S) (f : R →+* S) (hIJ : I ≤ J.comap f) : R ⧸ I →+* S ⧸ J :=
  Quotient.lift I ((Quotient.mk J).comp f) fun _ ha => by
    /-
      R : Type u
      S✝ : Type v
      F : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : Semiring S✝
      R₁ : Type u_1
      R₂ : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁵ : CommSemiring R₁
      inst✝⁴ : CommSemiring R₂
      inst✝³ : CommRing A
      inst✝² : Algebra R₁ A
      inst✝¹ : Algebra R₂ A
      S : Type v
      inst✝ : CommRing S
      I : Ideal R
      J : Ideal S
      f : RingHom R S
      hIJ : LE.le I (Ideal.comap f J)
      x✝ : R
      ha : Membership.mem I x✝
      ⊢ Eq (((Ideal.Quotient.mk J).comp f) x✝) 0
    -/
    simpa [Function.comp_apply, RingHom.coe_comp, Quotient.eq_zero_iff_mem] using hIJ ha
    /-
      🎉 no goals
    -/


@[simp]
theorem quotientMap_mk {J : Ideal R} {I : Ideal S} {f : R →+* S} {H : J ≤ I.comap f} {x : R} :
    quotientMap I f H (Quotient.mk J x) = Quotient.mk I (f x) :=
  Quotient.lift_mk J _ _


@[simp]
theorem quotientMap_algebraMap {J : Ideal A} {I : Ideal S} {f : A →+* S} {H : J ≤ I.comap f}
    {x : R₁} : quotientMap I f H (algebraMap R₁ (A ⧸ J) x) = Quotient.mk I (f (algebraMap _ _ x)) :=
  Quotient.lift_mk J _ _


theorem quotientMap_comp_mk {J : Ideal R} {I : Ideal S} {f : R →+* S} (H : J ≤ I.comap f) :
    (quotientMap I f H).comp (Quotient.mk J) = (Quotient.mk I).comp f :=
                          /-
                            R : Type u
                            inst✝¹ : CommRing R
                            S : Type v
                            inst✝ : CommRing S
                            J : Ideal R
                            I : Ideal S
                            f : RingHom R S
                            H : LE.le J (Ideal.comap f I)
                            x : R
                            ⊢ Eq (((Ideal.quotientMap I f H).comp (Ideal.Quotient.mk J)) x) (((Ideal.Quoti …
                          -/
  RingHom.ext fun x => by simp only [Function.comp_apply, RingHom.coe_comp, Ideal.quotientMap_mk]
                          /-
                            🎉 no goals
                          -/


lemma ker_quotientMap_mk {I J : Ideal R} :
    RingHom.ker (quotientMap (J.map _) (Quotient.mk I) le_comap_map) = I.map (Quotient.mk J) := by
  rw [Ideal.quotientMap, Ideal.ker_quotient_lift, ← RingHom.comap_ker, Ideal.mk_ker,
    Ideal.comap_map_of_surjective _ Ideal.Quotient.mk_surjective,
    ← RingHom.ker_eq_comap_bot, Ideal.mk_ker, Ideal.map_sup, Ideal.map_quotient_self, bot_sup_eq]


/-- The ring equiv `R/I ≃+* S/J` induced by a ring equiv `f : R ≃+* S`, where `J = f(I)`. -/
@[simps]
def quotientEquiv (I : Ideal R) (J : Ideal S) (f : R ≃+* S) (hIJ : J = I.map (f : R →+* S)) :
    R ⧸ I ≃+* S ⧸ J :=
  {
    quotientMap J (↑f) (by
      /-
        R : Type u
        S✝ : Type v
        F : Type w
        inst✝⁷ : CommRing R
        inst✝⁶ : Semiring S✝
        R₁ : Type u_1
        R₂ : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁵ : CommSemiring R₁
        inst✝⁴ : CommSemiring R₂
        inst✝³ : CommRing A
        inst✝² : Algebra R₁ A
        inst✝¹ : Algebra R₂ A
        S : Type v
        inst✝ : CommRing S
        I : Ideal R
        J : Ideal S
        f : RingEquiv R S
        hIJ : Eq J (Ideal.map (↑f) I)
        ⊢ LE.le I (Ideal.comap (↑f) J)
      -/
      rw [hIJ]
      /-
        R : Type u
        S✝ : Type v
        F : Type w
        inst✝⁷ : CommRing R
        inst✝⁶ : Semiring S✝
        R₁ : Type u_1
        R₂ : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁵ : CommSemiring R₁
        inst✝⁴ : CommSemiring R₂
        inst✝³ : CommRing A
        inst✝² : Algebra R₁ A
        inst✝¹ : Algebra R₂ A
        S : Type v
        inst✝ : CommRing S
        I : Ideal R
        J : Ideal S
        f : RingEquiv R S
        hIJ : Eq J (Ideal.map (↑f) I)
        ⊢ LE.le I (Ideal.comap (↑f) (Ideal.map (↑f) I))
      -/
      exact le_comap_map)
      /-
        🎉 no goals
      -/
  with
    invFun :=
      quotientMap I (↑f.symm)
        (by
          /-
            R : Type u
            S✝ : Type v
            F : Type w
            inst✝⁷ : CommRing R
            inst✝⁶ : Semiring S✝
            R₁ : Type u_1
            R₂ : Type u_2
            A : Type u_3
            B : Type u_4
            inst✝⁵ : CommSemiring R₁
            inst✝⁴ : CommSemiring R₂
            inst✝³ : CommRing A
            inst✝² : Algebra R₁ A
            inst✝¹ : Algebra R₂ A
            S : Type v
            inst✝ : CommRing S
            I : Ideal R
            J : Ideal S
            f : RingEquiv R S
            hIJ : Eq J (Ideal.map (↑f) I)
            ⊢ LE.le J (Ideal.comap (↑f.symm) I)
          -/
          rw [hIJ]
          /-
            R : Type u
            S✝ : Type v
            F : Type w
            inst✝⁷ : CommRing R
            inst✝⁶ : Semiring S✝
            R₁ : Type u_1
            R₂ : Type u_2
            A : Type u_3
            B : Type u_4
            inst✝⁵ : CommSemiring R₁
            inst✝⁴ : CommSemiring R₂
            inst✝³ : CommRing A
            inst✝² : Algebra R₁ A
            inst✝¹ : Algebra R₂ A
            S : Type v
            inst✝ : CommRing S
            I : Ideal R
            J : Ideal S
            f : RingEquiv R S
            hIJ : Eq J (Ideal.map (↑f) I)
            ⊢ LE.le (Ideal.map (↑f) I) (Ideal.comap (↑f.symm) I)
          -/
          exact le_of_eq (map_comap_of_equiv f))
          /-
            🎉 no goals
          -/
    left_inv := by
      /-
        R : Type u
        S✝ : Type v
        F : Type w
        inst✝⁷ : CommRing R
        inst✝⁶ : Semiring S✝
        R₁ : Type u_1
        R₂ : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁵ : CommSemiring R₁
        inst✝⁴ : CommSemiring R₂
        inst✝³ : CommRing A
        inst✝² : Algebra R₁ A
        inst✝¹ : Algebra R₂ A
        S : Type v
        inst✝ : CommRing S
        I : Ideal R
        J : Ideal S
        f : RingEquiv R S
        hIJ : Eq J (Ideal.map (↑f) I)
        ⊢ Function.LeftInverse (⇑(Ideal.quotientMap I ↑f.symm ⋯)) (↑↑__src✝).toFun
      -/
      rintro ⟨r⟩
      simp only [Submodule.Quotient.quot_mk_eq_mk, Quotient.mk_eq_mk, RingHom.toFun_eq_coe,
        quotientMap_mk, RingEquiv.coe_toRingHom, RingEquiv.symm_apply_apply]
    right_inv := by
      /-
        R : Type u
        S✝ : Type v
        F : Type w
        inst✝⁷ : CommRing R
        inst✝⁶ : Semiring S✝
        R₁ : Type u_1
        R₂ : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁵ : CommSemiring R₁
        inst✝⁴ : CommSemiring R₂
        inst✝³ : CommRing A
        inst✝² : Algebra R₁ A
        inst✝¹ : Algebra R₂ A
        S : Type v
        inst✝ : CommRing S
        I : Ideal R
        J : Ideal S
        f : RingEquiv R S
        hIJ : Eq J (Ideal.map (↑f) I)
        ⊢ Function.RightInverse (⇑(Ideal.quotientMap I ↑f.symm ⋯)) (↑↑__src✝).toFun
      -/
      rintro ⟨s⟩
      simp only [Submodule.Quotient.quot_mk_eq_mk, Quotient.mk_eq_mk, RingHom.toFun_eq_coe,
        quotientMap_mk, RingEquiv.coe_toRingHom, RingEquiv.apply_symm_apply] }

/- Porting note: removed simp. LHS simplified. Slightly different version of the simplified
form closed this and was itself closed by simp -/

theorem quotientEquiv_mk (I : Ideal R) (J : Ideal S) (f : R ≃+* S) (hIJ : J = I.map (f : R →+* S))
    (x : R) : quotientEquiv I J f hIJ (Ideal.Quotient.mk I x) = Ideal.Quotient.mk J (f x) :=
  rfl


@[simp]
theorem quotientEquiv_symm_mk (I : Ideal R) (J : Ideal S) (f : R ≃+* S)
    (hIJ : J = I.map (f : R →+* S)) (x : S) :
    (quotientEquiv I J f hIJ).symm (Ideal.Quotient.mk J x) = Ideal.Quotient.mk I (f.symm x) :=
  rfl


/-- `H` and `h` are kept as separate hypothesis since H is used in constructing the quotient map. -/
theorem quotientMap_injective' {J : Ideal R} {I : Ideal S} {f : R →+* S} {H : J ≤ I.comap f}
    (h : I.comap f ≤ J) : Function.Injective (quotientMap I f H) := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    J : Ideal R
    I : Ideal S
    f : RingHom R S
    H : LE.le J (Ideal.comap f I)
    h : LE.le (Ideal.comap f I) J
    ⊢ Function.Injective ⇑(Ideal.quotientMap I f H)
  -/
  refine (injective_iff_map_eq_zero (quotientMap I f H)).2 fun a ha => ?_
  /-
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    J : Ideal R
    I : Ideal S
    f : RingHom R S
    H : LE.le J (Ideal.comap f I)
    h : LE.le (Ideal.comap f I) J
    a : HasQuotient.Quotient R J
    ha : Eq ((Ideal.quotientMap I f H) a) 0
    ⊢ Eq a 0
  -/
  obtain ⟨r, rfl⟩ := Quotient.mk_surjective a
  /-
    case intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    J : Ideal R
    I : Ideal S
    f : RingHom R S
    H : LE.le J (Ideal.comap f I)
    h : LE.le (Ideal.comap f I) J
    r : R
    ha : Eq ((Ideal.quotientMap I f H) ((Ideal.Quotient.mk J) r)) 0
    ⊢ Eq ((Ideal.Quotient.mk J) r) 0
  -/
  rw [quotientMap_mk, Quotient.eq_zero_iff_mem] at ha
  /-
    case intro
    R : Type u
    inst✝¹ : CommRing R
    S : Type v
    inst✝ : CommRing S
    J : Ideal R
    I : Ideal S
    f : RingHom R S
    H : LE.le J (Ideal.comap f I)
    h : LE.le (Ideal.comap f I) J
    r : R
    ha : Membership.mem I (f r)
    ⊢ Eq ((Ideal.Quotient.mk J) r) 0
  -/
  exact Quotient.eq_zero_iff_mem.mpr (h ha)
  /-
    🎉 no goals
  -/


/-- If we take `J = I.comap f` then `quotientMap` is injective automatically. -/
theorem quotientMap_injective {I : Ideal S} {f : R →+* S} :
    Function.Injective (quotientMap I f le_rfl) :=
  quotientMap_injective' le_rfl


theorem quotientMap_surjective {J : Ideal R} {I : Ideal S} {f : R →+* S} {H : J ≤ I.comap f}
    (hf : Function.Surjective f) : Function.Surjective (quotientMap I f H) := fun x =>
  let ⟨x, hx⟩ := Quotient.mk_surjective x
  let ⟨y, hy⟩ := hf x
                         /-
                           R : Type u
                           inst✝¹ : CommRing R
                           S : Type v
                           inst✝ : CommRing S
                           J : Ideal R
                           I : Ideal S
                           f : RingHom R S
                           H : LE.le J (Ideal.comap f I)
                           hf : Function.Surjective ⇑f
                           x✝ : HasQuotient.Quotient S I
                           x : S
                           hx : Eq ((Ideal.Quotient.mk I) x) x✝
                           y : R
                           hy : Eq (f y) x
                           ⊢ Eq ((Ideal.quotientMap I f H) ((Ideal.Quotient.mk J) y)) x✝
                         -/
  ⟨(Quotient.mk J) y, by simp [hx, hy]⟩
                         /-
                           🎉 no goals
                         -/


/-- Commutativity of a square is preserved when taking quotients by an ideal. -/
theorem comp_quotientMap_eq_of_comp_eq {R' S' : Type*} [CommRing R'] [CommRing S'] {f : R →+* S}
    {f' : R' →+* S'} {g : R →+* R'} {g' : S →+* S'} (hfg : f'.comp g = g'.comp f) (I : Ideal S') :
    -- Porting note: was losing track of I
    let leq := le_of_eq (_root_.trans (comap_comap (I := I) f g') (hfg ▸ comap_comap (I := I) g f'))
    (quotientMap I g' le_rfl).comp (quotientMap (I.comap g') f le_rfl) =
    (quotientMap I f' le_rfl).comp (quotientMap (I.comap f') g leq) := by
  /-
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    R' : Type u_5
    S' : Type u_6
    inst✝¹ : CommRing R'
    inst✝ : CommRing S'
    f : RingHom R S
    f' : RingHom R' S'
    g : RingHom R R'
    g' : RingHom S S'
    hfg : Eq (f'.comp g) (g'.comp f)
    I : Ideal S'
    ⊢ let leq := ⋯;
      Eq ((Ideal.quotientMap I g' ⋯).comp (Ideal.quotientMap (Ideal.comap g' I) f  …
  -/
  refine RingHom.ext fun a => ?_
  /-
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    R' : Type u_5
    S' : Type u_6
    inst✝¹ : CommRing R'
    inst✝ : CommRing S'
    f : RingHom R S
    f' : RingHom R' S'
    g : RingHom R R'
    g' : RingHom S S'
    hfg : Eq (f'.comp g) (g'.comp f)
    I : Ideal S'
    a : HasQuotient.Quotient R (Ideal.comap f (Ideal.comap g' I))
    ⊢ Eq (((Ideal.quotientMap I g' ⋯).comp (Ideal.quotientMap (Ideal.comap g' I) f …
  -/
  obtain ⟨r, rfl⟩ := Quotient.mk_surjective a
  /-
    case intro
    R : Type u
    inst✝³ : CommRing R
    S : Type v
    inst✝² : CommRing S
    R' : Type u_5
    S' : Type u_6
    inst✝¹ : CommRing R'
    inst✝ : CommRing S'
    f : RingHom R S
    f' : RingHom R' S'
    g : RingHom R R'
    g' : RingHom S S'
    hfg : Eq (f'.comp g) (g'.comp f)
    I : Ideal S'
    r : R
    ⊢ Eq (((Ideal.quotientMap I g' ⋯).comp (Ideal.quotientMap (Ideal.comap g' I) f …
  -/
  simp only [RingHom.comp_apply, quotientMap_mk]
  exact (Ideal.Quotient.mk I).congr_arg (_root_.trans (g'.comp_apply f r).symm
    (hfg ▸ f'.comp_apply g r))


/-- The algebra hom `A/I →+* B/J` induced by an algebra hom `f : A →ₐ[R₁] B` with `I ≤ f⁻¹(J)`. -/
def quotientMapₐ {I : Ideal A} (J : Ideal B) (f : A →ₐ[R₁] B) (hIJ : I ≤ J.comap f) :
    A ⧸ I →ₐ[R₁] B ⧸ J :=
  { quotientMap J (f : A →+* B) hIJ with commutes' := fun r => by simp only [RingHom.toFun_eq_coe,
    quotientMap_algebraMap, AlgHom.coe_toRingHom, AlgHom.commutes, Quotient.mk_algebraMap] }


@[simp]
theorem quotient_map_mkₐ {I : Ideal A} (J : Ideal B) (f : A →ₐ[R₁] B) (H : I ≤ J.comap f) {x : A} :
    quotientMapₐ J f H (Quotient.mk I x) = Quotient.mkₐ R₁ J (f x) :=
  rfl


theorem quotient_map_comp_mkₐ {I : Ideal A} (J : Ideal B) (f : A →ₐ[R₁] B) (H : I ≤ J.comap f) :
    (quotientMapₐ J f H).comp (Quotient.mkₐ R₁ I) = (Quotient.mkₐ R₁ J).comp f :=
                         /-
                           R₁ : Type u_1
                           A : Type u_3
                           B : Type u_4
                           inst✝⁴ : CommSemiring R₁
                           inst✝³ : CommRing A
                           inst✝² : Algebra R₁ A
                           inst✝¹ : CommRing B
                           inst✝ : Algebra R₁ B
                           I : Ideal A
                           J : Ideal B
                           f : AlgHom R₁ A B
                           H : LE.le I (Ideal.comap f J)
                           x : A
                           ⊢ Eq (((Ideal.quotientMapₐ J f H).comp (Ideal.Quotient.mkₐ R₁ I)) x) (((Ideal. …
                         -/
  AlgHom.ext fun x => by simp only [quotient_map_mkₐ, Quotient.mkₐ_eq_mk, AlgHom.comp_apply]
                         /-
                           🎉 no goals
                         -/


/-- The algebra equiv `A/I ≃ₐ[R] B/J` induced by an algebra equiv `f : A ≃ₐ[R] B`,
where`J = f(I)`. -/
def quotientEquivAlg (I : Ideal A) (J : Ideal B) (f : A ≃ₐ[R₁] B) (hIJ : J = I.map (f : A →+* B)) :
    (A ⧸ I) ≃ₐ[R₁] B ⧸ J :=
  { quotientEquiv I J (f : A ≃+* B) hIJ with
    commutes' := fun r => by
      -- Porting note: Needed to add the below lemma because Equivs coerce weird
      have : ∀ (e : RingEquiv (A ⧸ I) (B ⧸ J)), Equiv.toFun e.toEquiv = DFunLike.coe e :=
        fun _ ↦ rfl
      /-
        R : Type u
        S : Type v
        F : Type w
        inst✝⁸ : CommRing R
        inst✝⁷ : Semiring S
        R₁ : Type u_1
        R₂ : Type u_2
        A : Type u_3
        B : Type u_4
        inst✝⁶ : CommSemiring R₁
        inst✝⁵ : CommSemiring R₂
        inst✝⁴ : CommRing A
        inst✝³ : Algebra R₁ A
        inst✝² : Algebra R₂ A
        inst✝¹ : CommRing B
        inst✝ : Algebra R₁ B
        I : Ideal A
        J : Ideal B
        f : AlgEquiv R₁ A B
        hIJ : Eq J (Ideal.map (↑f) I)
        r : R₁
        this : ∀ (e : RingEquiv (HasQuotient.Quotient A I) (HasQuotient.Quotient B J)) …
        ⊢ Eq (__src✝.toFun ((algebraMap R₁ (HasQuotient.Quotient A I)) r)) ((algebraMa …
      -/
      rw [this]
      simp only [quotientEquiv_apply, RingHom.toFun_eq_coe, quotientMap_algebraMap,
      RingEquiv.coe_toRingHom, AlgEquiv.coe_ringEquiv, AlgEquiv.commutes, Quotient.mk_algebraMap]}


/-- If `P` lies over `p`, then `R / p` has a canonical map to `A / P`. -/
abbrev Quotient.algebraQuotientOfLEComap [Algebra R A] {p : Ideal R} {P : Ideal A}
    (h : p ≤ comap (algebraMap R A) P) : Algebra (R ⧸ p) (A ⧸ P) where
  toRingHom := quotientMap P (algebraMap R A) h
  smul := Quotient.lift₂ (⟦· • ·⟧) fun r₁ a₁ r₂ a₂ hr ha ↦ Quotient.sound <| by
    /-
      R : Type u
      S : Type v
      F : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : Semiring S
      R₁ : Type u_1
      R₂ : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁵ : CommSemiring R₁
      inst✝⁴ : CommSemiring R₂
      inst✝³ : CommRing A
      inst✝² : Algebra R₁ A
      inst✝¹ : Algebra R₂ A
      inst✝ : Algebra R A
      p : Ideal R
      P : Ideal A
      h : LE.le p (Ideal.comap (algebraMap R A) P)
      r₁ : R
      a₁ : A
      r₂ : R
      a₂ : A
      hr : HasEquiv.Equiv r₁ r₂
      ha : HasEquiv.Equiv a₁ a₂
      ⊢ HasEquiv.Equiv (HSMul.hSMul r₁ a₁) (HSMul.hSMul r₂ a₂)
    -/
    have := h (p.quotientRel_def.mp hr)
    /-
      R : Type u
      S : Type v
      F : Type w
      inst✝⁷ : CommRing R
      inst✝⁶ : Semiring S
      R₁ : Type u_1
      R₂ : Type u_2
      A : Type u_3
      B : Type u_4
      inst✝⁵ : CommSemiring R₁
      inst✝⁴ : CommSemiring R₂
      inst✝³ : CommRing A
      inst✝² : Algebra R₁ A
      inst✝¹ : Algebra R₂ A
      inst✝ : Algebra R A
      p : Ideal R
      P : Ideal A
      h : LE.le p (Ideal.comap (algebraMap R A) P)
      r₁ : R
      a₁ : A
      r₂ : R
      a₂ : A
      hr : HasEquiv.Equiv r₁ r₂
      ha : HasEquiv.Equiv a₁ a₂
      this : Membership.mem (Ideal.comap (algebraMap R A) P) (HSub.hSub r₁ r₂)
      ⊢ HasEquiv.Equiv (HSMul.hSMul r₁ a₁) (HSMul.hSMul r₂ a₂)
    -/
    rw [mem_comap, map_sub] at this
    simpa only [Algebra.smul_def] using P.quotientRel_def.mpr
      (P.mul_sub_mul_mem this <| P.quotientRel_def.mp ha)
                  /-
                    R : Type u
                    S : Type v
                    F : Type w
                    inst✝⁷ : CommRing R
                    inst✝⁶ : Semiring S
                    R₁ : Type u_1
                    R₂ : Type u_2
                    A : Type u_3
                    B : Type u_4
                    inst✝⁵ : CommSemiring R₁
                    inst✝⁴ : CommSemiring R₂
                    inst✝³ : CommRing A
                    inst✝² : Algebra R₁ A
                    inst✝¹ : Algebra R₂ A
                    inst✝ : Algebra R A
                    p : Ideal R
                    P : Ideal A
                    h : LE.le p (Ideal.comap (algebraMap R A) P)
                    ⊢ ∀ (r : HasQuotient.Quotient R p) (x : HasQuotient.Quotient A P), Eq (HSMul.h …
                  -/
                  /-
                    R : Type u
                    S : Type v
                    F : Type w
                    inst✝⁷ : CommRing R
                    inst✝⁶ : Semiring S
                    R₁ : Type u_1
                    R₂ : Type u_2
                    A : Type u_3
                    B : Type u_4
                    inst✝⁵ : CommSemiring R₁
                    inst✝⁴ : CommSemiring R₂
                    inst✝³ : CommRing A
                    inst✝² : Algebra R₁ A
                    inst✝¹ : Algebra R₂ A
                    inst✝ : Algebra R A
                    p : Ideal R
                    P : Ideal A
                    h : LE.le p (Ideal.comap (algebraMap R A) P)
                    ⊢ ∀ (r : HasQuotient.Quotient R p) (x : HasQuotient.Quotient A P), Eq (HMul.hM …
                  -/
  smul_def' := by rintro ⟨_⟩ ⟨_⟩; exact congr_arg (⟦·⟧) (Algebra.smul_def _ _)
                                  /-
                                    🎉 no goals
                                  -/
                                  /-
                                    🎉 no goals
                                  -/
  commutes' := by rintro ⟨_⟩ ⟨_⟩; exact congr_arg (⟦·⟧) (Algebra.commutes _ _)


instance (priority := 100) quotientAlgebra {I : Ideal A} [Algebra R A] :
    Algebra (R ⧸ I.comap (algebraMap R A)) (A ⧸ I) :=
  Quotient.algebraQuotientOfLEComap le_rfl


theorem algebraMap_quotient_injective {I : Ideal A} [Algebra R A] :
    Function.Injective (algebraMap (R ⧸ I.comap (algebraMap R A)) (A ⧸ I)) := by
  /-
    R : Type u
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : CommRing A
    I : Ideal A
    inst✝ : Algebra R A
    ⊢ Function.Injective ⇑(algebraMap (HasQuotient.Quotient R (Ideal.comap (algebr …
  -/
  rintro ⟨a⟩ ⟨b⟩ hab
  /-
    case mk.mk
    R : Type u
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : CommRing A
    I : Ideal A
    inst✝ : Algebra R A
    a₁✝ : HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I)
    a : R
    a₂✝ : HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I)
    b : R
    hab : Eq ((algebraMap (HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I) …
    ⊢ Eq (Quot.mk (⇑(Submodule.quotientRel (Ideal.comap (algebraMap R A) I))) a) ( …
  -/
  replace hab := Quotient.eq.mp hab
  /-
    case mk.mk
    R : Type u
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : CommRing A
    I : Ideal A
    inst✝ : Algebra R A
    a₁✝ : HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I)
    a : R
    a₂✝ : HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I)
    b : R
    hab : Membership.mem I (HSub.hSub ((algebraMap R A) a) ((algebraMap R A) b))
    ⊢ Eq (Quot.mk (⇑(Submodule.quotientRel (Ideal.comap (algebraMap R A) I))) a) ( …
  -/
  rw [← RingHom.map_sub] at hab
  /-
    case mk.mk
    R : Type u
    inst✝² : CommRing R
    A : Type u_3
    inst✝¹ : CommRing A
    I : Ideal A
    inst✝ : Algebra R A
    a₁✝ : HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I)
    a : R
    a₂✝ : HasQuotient.Quotient R (Ideal.comap (algebraMap R A) I)
    b : R
    hab : Membership.mem I ((algebraMap R A) (HSub.hSub a b))
    ⊢ Eq (Quot.mk (⇑(Submodule.quotientRel (Ideal.comap (algebraMap R A) I))) a) ( …
  -/
  exact Quotient.eq.mpr hab
  /-
    🎉 no goals
  -/


/-- Quotienting by equal ideals gives equivalent algebras. -/
def quotientEquivAlgOfEq {I J : Ideal A} (h : I = J) : (A ⧸ I) ≃ₐ[R₁] A ⧸ J :=
  quotientEquivAlg I J AlgEquiv.refl <| h ▸ (map_id I).symm


@[simp]
theorem quotientEquivAlgOfEq_mk {I J : Ideal A} (h : I = J) (x : A) :
    quotientEquivAlgOfEq R₁ h (Ideal.Quotient.mk I x) = Ideal.Quotient.mk J x :=
  rfl


@[simp]
theorem quotientEquivAlgOfEq_symm {I J : Ideal A} (h : I = J) :
    (quotientEquivAlgOfEq R₁ h).symm = quotientEquivAlgOfEq R₁ h.symm := by
  /-
    R₁ : Type u_1
    A : Type u_3
    inst✝² : CommSemiring R₁
    inst✝¹ : CommRing A
    inst✝ : Algebra R₁ A
    I J : Ideal A
    h : Eq I J
    ⊢ Eq (Ideal.quotientEquivAlgOfEq R₁ h).symm (Ideal.quotientEquivAlgOfEq R₁ ⋯)
  -/
  ext
  /-
    case h
    R₁ : Type u_1
    A : Type u_3
    inst✝² : CommSemiring R₁
    inst✝¹ : CommRing A
    inst✝ : Algebra R₁ A
    I J : Ideal A
    h : Eq I J
    a✝ : HasQuotient.Quotient A J
    ⊢ Eq ((Ideal.quotientEquivAlgOfEq R₁ h).symm a✝) ((Ideal.quotientEquivAlgOfEq  …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma comap_map_mk {I J : Ideal R} (h : I ≤ J) :
    Ideal.comap (Ideal.Quotient.mk I) (Ideal.map (Ideal.Quotient.mk I) J) = J := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    ⊢ Eq (Ideal.comap (Ideal.Quotient.mk I) (Ideal.map (Ideal.Quotient.mk I) J)) J
  -/
  ext; rw [← Ideal.mem_quotient_iff_mem h, Ideal.mem_comap]
       /-
         🎉 no goals
       -/


/-- The **first isomorphism theorem** for commutative algebras (`AlgHom.range` version). -/
noncomputable def quotientKerEquivRange
  {A B : Type*} [CommRing A] [Algebra R A] [Semiring B] [Algebra R B]
  (f : A →ₐ[R] B) :
  (A ⧸ RingHom.ker f) ≃ₐ[R] f.range :=
  (Ideal.quotientEquivAlgOfEq R (AlgHom.ker_rangeRestrict f).symm).trans <|
    Ideal.quotientKerAlgEquivOfSurjective f.rangeRestrict_surjective


/-- The obvious ring hom `R/I → R/(I ⊔ J)` -/
def quotLeftToQuotSup : R ⧸ I →+* R ⧸ I ⊔ J :=
  Ideal.Quotient.factor I (I ⊔ J) le_sup_left


/-- The kernel of `quotLeftToQuotSup` -/
theorem ker_quotLeftToQuotSup : RingHom.ker (quotLeftToQuotSup I J) =
    J.map (Ideal.Quotient.mk I) := by
  simp only [mk_ker, sup_idem, sup_comm, quotLeftToQuotSup, Quotient.factor, ker_quotient_lift,
    map_eq_iff_sup_ker_eq_of_surjective (Ideal.Quotient.mk I) Quotient.mk_surjective, ← sup_assoc]


/-- The ring homomorphism `(R/I)/J' -> R/(I ⊔ J)` induced by `quotLeftToQuotSup` where `J'`
  is the image of `J` in `R/I`-/
def quotQuotToQuotSup : (R ⧸ I) ⧸ J.map (Ideal.Quotient.mk I) →+* R ⧸ I ⊔ J :=
  Ideal.Quotient.lift (J.map (Ideal.Quotient.mk I)) (quotLeftToQuotSup I J)
    (ker_quotLeftToQuotSup I J).symm.le


/-- The composite of the maps `R → (R/I)` and `(R/I) → (R/I)/J'` -/
def quotQuotMk : R →+* (R ⧸ I) ⧸ J.map (Ideal.Quotient.mk I) :=
  (Ideal.Quotient.mk (J.map (Ideal.Quotient.mk I))).comp (Ideal.Quotient.mk I)

-- Porting note: mismatched instances

/-- The kernel of `quotQuotMk` -/
theorem ker_quotQuotMk : RingHom.ker (quotQuotMk I J) = I ⊔ J := by
  rw [RingHom.ker_eq_comap_bot, quotQuotMk, ← comap_comap, ← RingHom.ker, mk_ker,
    comap_map_of_surjective (Ideal.Quotient.mk I) Ideal.Quotient.mk_surjective, ← RingHom.ker,
    mk_ker, sup_comm]


/-- The ring homomorphism `R/(I ⊔ J) → (R/I)/J' `induced by `quotQuotMk` -/
def liftSupQuotQuotMk (I J : Ideal R) : R ⧸ I ⊔ J →+* (R ⧸ I) ⧸ J.map (Ideal.Quotient.mk I) :=
  Ideal.Quotient.lift (I ⊔ J) (quotQuotMk I J) (ker_quotQuotMk I J).symm.le


/-- `quotQuotToQuotSup` and `liftSupQuotQuotMk` are inverse isomorphisms. In the case where
    `I ≤ J`, this is the Third Isomorphism Theorem (see `quotQuotEquivQuotOfLe`)-/
def quotQuotEquivQuotSup : (R ⧸ I) ⧸ J.map (Ideal.Quotient.mk I) ≃+* R ⧸ I ⊔ J :=
  RingEquiv.ofHomInv (quotQuotToQuotSup I J) (liftSupQuotQuotMk I J)
    (by
      /-
        R : Type u
        inst✝ : CommRing R
        I J : Ideal R
        ⊢ Eq ((↑(DoubleQuot.liftSupQuotQuotMk I J)).comp ↑(DoubleQuot.quotQuotToQuotSu …
      -/
      repeat apply Ideal.Quotient.ringHom_ext
      /-
        case h.h
        R : Type u
        inst✝ : CommRing R
        I J : Ideal R
        ⊢ Eq ((((↑(DoubleQuot.liftSupQuotQuotMk I J)).comp ↑(DoubleQuot.quotQuotToQuot …
      -/
      rfl)
      /-
        🎉 no goals
      -/
    (by
      /-
        R : Type u
        inst✝ : CommRing R
        I J : Ideal R
        ⊢ Eq ((↑(DoubleQuot.quotQuotToQuotSup I J)).comp ↑(DoubleQuot.liftSupQuotQuotM …
      -/
      repeat apply Ideal.Quotient.ringHom_ext
      /-
        case h
        R : Type u
        inst✝ : CommRing R
        I J : Ideal R
        ⊢ Eq (((↑(DoubleQuot.quotQuotToQuotSup I J)).comp ↑(DoubleQuot.liftSupQuotQuot …
      -/
      rfl)
      /-
        🎉 no goals
      -/


@[simp]
theorem quotQuotEquivQuotSup_quotQuotMk (x : R) :
    quotQuotEquivQuotSup I J (quotQuotMk I J x) = Ideal.Quotient.mk (I ⊔ J) x :=
  rfl


@[simp]
theorem quotQuotEquivQuotSup_symm_quotQuotMk (x : R) :
    (quotQuotEquivQuotSup I J).symm (Ideal.Quotient.mk (I ⊔ J) x) = quotQuotMk I J x :=
  rfl


/-- The obvious isomorphism `(R/I)/J' → (R/J)/I'` -/
def quotQuotEquivComm : (R ⧸ I) ⧸ J.map (Ideal.Quotient.mk I) ≃+*
    (R ⧸ J) ⧸ I.map (Ideal.Quotient.mk J) :=
  ((quotQuotEquivQuotSup I J).trans (quotEquivOfEq (sup_comm ..))).trans
    (quotQuotEquivQuotSup J I).symm

-- Porting note: mismatched instances

@[simp]
theorem quotQuotEquivComm_quotQuotMk (x : R) :
    quotQuotEquivComm I J (quotQuotMk I J x) = quotQuotMk J I x :=
  rfl

-- Porting note: mismatched instances

@[simp]
theorem quotQuotEquivComm_comp_quotQuotMk :
    RingHom.comp (↑(quotQuotEquivComm I J)) (quotQuotMk I J) = quotQuotMk J I :=
  RingHom.ext <| quotQuotEquivComm_quotQuotMk I J


@[simp]
theorem quotQuotEquivComm_symm : (quotQuotEquivComm I J).symm = quotQuotEquivComm J I := by
  /-  Porting note: this proof used to just be rfl but currently rfl opens up a bottomless pit
  of processor cycles. Synthesizing instances does not seem to be an issue.
  -/
  change (((quotQuotEquivQuotSup I J).trans (quotEquivOfEq (sup_comm ..))).trans
    (quotQuotEquivQuotSup J I).symm).symm =
      ((quotQuotEquivQuotSup J I).trans (quotEquivOfEq (sup_comm ..))).trans
        (quotQuotEquivQuotSup I J).symm
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    ⊢ Eq (((DoubleQuot.quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfEq ⋯)).tr …
  -/
  ext r
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    r : HasQuotient.Quotient (HasQuotient.Quotient R J) (Ideal.map (Ideal.Quotient …
    ⊢ Eq ((((DoubleQuot.quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfEq ⋯)).t …
  -/
  dsimp
  /-
    case h
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    r : HasQuotient.Quotient (HasQuotient.Quotient R J) (Ideal.map (Ideal.Quotient …
    ⊢ Eq ((DoubleQuot.quotQuotEquivQuotSup I J).symm ((Ideal.quotEquivOfEq ⋯).symm …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- **The Third Isomorphism theorem** for rings. See `quotQuotEquivQuotSup` for a version
    that does not assume an inclusion of ideals. -/
def quotQuotEquivQuotOfLE (h : I ≤ J) : (R ⧸ I) ⧸ J.map (Ideal.Quotient.mk I) ≃+* R ⧸ J :=
  (quotQuotEquivQuotSup I J).trans (Ideal.quotEquivOfEq <| sup_eq_right.mpr h)


@[simp]
theorem quotQuotEquivQuotOfLE_quotQuotMk (x : R) (h : I ≤ J) :
    quotQuotEquivQuotOfLE h (quotQuotMk I J x) = (Ideal.Quotient.mk J) x :=
  rfl


@[simp]
theorem quotQuotEquivQuotOfLE_symm_mk (x : R) (h : I ≤ J) :
    (quotQuotEquivQuotOfLE h).symm ((Ideal.Quotient.mk J) x) = quotQuotMk I J x :=
  rfl


theorem quotQuotEquivQuotOfLE_comp_quotQuotMk (h : I ≤ J) :
    RingHom.comp (↑(quotQuotEquivQuotOfLE h)) (quotQuotMk I J) = (Ideal.Quotient.mk J) := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    ⊢ Eq ((↑(DoubleQuot.quotQuotEquivQuotOfLE h)).comp (DoubleQuot.quotQuotMk I J) …
  -/
  ext
  /-
    case a
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    x✝ : R
    ⊢ Eq (((↑(DoubleQuot.quotQuotEquivQuotOfLE h)).comp (DoubleQuot.quotQuotMk I J …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem quotQuotEquivQuotOfLE_symm_comp_mk (h : I ≤ J) :
    RingHom.comp (↑(quotQuotEquivQuotOfLE h).symm) (Ideal.Quotient.mk J) = quotQuotMk I J := by
  /-
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    ⊢ Eq ((↑(DoubleQuot.quotQuotEquivQuotOfLE h).symm).comp (Ideal.Quotient.mk J)) …
  -/
  ext
  /-
    case a
    R : Type u
    inst✝ : CommRing R
    I J : Ideal R
    h : LE.le I J
    x✝ : R
    ⊢ Eq (((↑(DoubleQuot.quotQuotEquivQuotOfLE h).symm).comp (Ideal.Quotient.mk J) …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem quotQuotEquivComm_mk_mk [CommRing R] (I J : Ideal R) (x : R) :
    quotQuotEquivComm I J (Ideal.Quotient.mk _ (Ideal.Quotient.mk _ x)) = algebraMap R _ x :=
  rfl


@[simp]
theorem quotQuotEquivQuotSup_quot_quot_algebraMap (x : R) :
    DoubleQuot.quotQuotEquivQuotSup I J (algebraMap R _ x) = algebraMap _ _ x :=
  rfl


@[simp]
theorem quotQuotEquivComm_algebraMap (x : R) :
    quotQuotEquivComm I J (algebraMap R _ x) = algebraMap _ _ x :=
  rfl


/-- The natural algebra homomorphism `A / I → A / (I ⊔ J)`. -/
def quotLeftToQuotSupₐ : A ⧸ I →ₐ[R] A ⧸ I ⊔ J :=
  AlgHom.mk (quotLeftToQuotSup I J) fun _ => rfl


@[simp]
theorem quotLeftToQuotSupₐ_toRingHom :
    (quotLeftToQuotSupₐ R I J : _ →+* _) = quotLeftToQuotSup I J :=
  rfl


@[simp]
theorem coe_quotLeftToQuotSupₐ : ⇑(quotLeftToQuotSupₐ R I J) = quotLeftToQuotSup I J :=
  rfl


/-- The algebra homomorphism `(A / I) / J' -> A / (I ⊔ J)` induced by `quotQuotToQuotSup`,
  where `J'` is the projection of `J` in `A / I`. -/
def quotQuotToQuotSupₐ : (A ⧸ I) ⧸ J.map (Quotient.mkₐ R I) →ₐ[R] A ⧸ I ⊔ J :=
  AlgHom.mk (quotQuotToQuotSup I J) fun _ => rfl


@[simp]
theorem quotQuotToQuotSupₐ_toRingHom :
    ((quotQuotToQuotSupₐ R I J) : _ ⧸ map (Ideal.Quotient.mkₐ R I) J →+* _) =
      quotQuotToQuotSup I J :=
  rfl


@[simp]
theorem coe_quotQuotToQuotSupₐ : ⇑(quotQuotToQuotSupₐ R I J) = quotQuotToQuotSup I J :=
  rfl


/-- The composition of the algebra homomorphisms `A → (A / I)` and `(A / I) → (A / I) / J'`,
  where `J'` is the projection `J` in `A / I`. -/
def quotQuotMkₐ : A →ₐ[R] (A ⧸ I) ⧸ J.map (Quotient.mkₐ R I) :=
  AlgHom.mk (quotQuotMk I J) fun _ => rfl


@[simp]
theorem quotQuotMkₐ_toRingHom :
    (quotQuotMkₐ R I J : _ →+* _ ⧸ J.map (Quotient.mkₐ R I)) = quotQuotMk I J :=
  rfl


@[simp]
theorem coe_quotQuotMkₐ : ⇑(quotQuotMkₐ R I J) = quotQuotMk I J :=
  rfl


/-- The injective algebra homomorphism `A / (I ⊔ J) → (A / I) / J'`induced by `quot_quot_mk`,
  where `J'` is the projection `J` in `A / I`. -/
def liftSupQuotQuotMkₐ (I J : Ideal A) : A ⧸ I ⊔ J →ₐ[R] (A ⧸ I) ⧸ J.map (Quotient.mkₐ R I) :=
  AlgHom.mk (liftSupQuotQuotMk I J) fun _ => rfl


@[simp]
theorem liftSupQuotQuotMkₐ_toRingHom :
    (liftSupQuotQuotMkₐ R I J : _ →+* _ ⧸ J.map (Quotient.mkₐ R I)) = liftSupQuotQuotMk I J :=
  rfl


@[simp]
theorem coe_liftSupQuotQuotMkₐ : ⇑(liftSupQuotQuotMkₐ R I J) = liftSupQuotQuotMk I J :=
  rfl


/-- `quotQuotToQuotSup` and `liftSupQuotQuotMk` are inverse isomorphisms. In the case where
`I ≤ J`, this is the Third Isomorphism Theorem (see `DoubleQuot.quotQuotEquivQuotOfLE`). -/
def quotQuotEquivQuotSupₐ : ((A ⧸ I) ⧸ J.map (Quotient.mkₐ R I)) ≃ₐ[R] A ⧸ I ⊔ J :=
  AlgEquiv.ofRingEquiv (f := quotQuotEquivQuotSup I J) fun _ => rfl


@[simp]
theorem quotQuotEquivQuotSupₐ_toRingEquiv :
    (quotQuotEquivQuotSupₐ R I J : _ ⧸ J.map (Quotient.mkₐ R I) ≃+* _) = quotQuotEquivQuotSup I J :=
  rfl


@[simp]
-- Porting note: had to add an extra coercion arrow on the right hand side.
theorem coe_quotQuotEquivQuotSupₐ : ⇑(quotQuotEquivQuotSupₐ R I J) = ⇑(quotQuotEquivQuotSup I J) :=
  rfl


@[simp]
theorem quotQuotEquivQuotSupₐ_symm_toRingEquiv :
    ((quotQuotEquivQuotSupₐ R I J).symm : _ ≃+* _ ⧸ J.map (Quotient.mkₐ R I)) =
      (quotQuotEquivQuotSup I J).symm :=
  rfl


@[simp]
-- Porting note: had to add an extra coercion arrow on the right hand side.
theorem coe_quotQuotEquivQuotSupₐ_symm :
    ⇑(quotQuotEquivQuotSupₐ R I J).symm = ⇑(quotQuotEquivQuotSup I J).symm :=
  rfl


/-- The natural algebra isomorphism `(A / I) / J' → (A / J) / I'`,
  where `J'` (resp. `I'`) is the projection of `J` in `A / I` (resp. `I` in `A / J`). -/
def quotQuotEquivCommₐ :
    ((A ⧸ I) ⧸ J.map (Quotient.mkₐ R I)) ≃ₐ[R] (A ⧸ J) ⧸ I.map (Quotient.mkₐ R J) :=
  AlgEquiv.ofRingEquiv (f := quotQuotEquivComm I J) fun _ => rfl


@[simp]
theorem quotQuotEquivCommₐ_toRingEquiv :
    (quotQuotEquivCommₐ R I J : _ ⧸ J.map (Quotient.mkₐ R I) ≃+* _ ⧸ I.map (Quotient.mkₐ R J)) =
      quotQuotEquivComm I J :=
  -- Porting note: should just be `rfl` but `AlgEquiv.toRingEquiv` and `AlgEquiv.ofRingEquiv`
  -- involve repacking everything in the structure, so Lean ends up unfolding `quotQuotEquivComm`
  -- and timing out.
  RingEquiv.ext fun _ => rfl


@[simp]
theorem coe_quotQuotEquivCommₐ : ⇑(quotQuotEquivCommₐ R I J) = ⇑(quotQuotEquivComm I J) :=
  rfl


@[simp]
theorem quotQuotEquivComm_symmₐ : (quotQuotEquivCommₐ R I J).symm = quotQuotEquivCommₐ R J I := by
  -- Porting note: should just be `rfl` but `AlgEquiv.toRingEquiv` and `AlgEquiv.ofRingEquiv`
  -- involve repacking everything in the structure, so Lean ends up unfolding `quotQuotEquivComm`
  -- and timing out.
  /-
    R : Type u
    A : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    I J : Ideal A
    ⊢ Eq (DoubleQuot.quotQuotEquivCommₐ R I J).symm (DoubleQuot.quotQuotEquivCommₐ …
  -/
  ext
  /-
    case h
    R : Type u
    A : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    I J : Ideal A
    a✝ : HasQuotient.Quotient (HasQuotient.Quotient A J) (Ideal.map (Ideal.Quotien …
    ⊢ Eq ((DoubleQuot.quotQuotEquivCommₐ R I J).symm a✝) ((DoubleQuot.quotQuotEqui …
  -/
  unfold quotQuotEquivCommₐ
  /-
    case h
    R : Type u
    A : Type u_1
    inst✝² : CommSemiring R
    inst✝¹ : CommRing A
    inst✝ : Algebra R A
    I J : Ideal A
    a✝ : HasQuotient.Quotient (HasQuotient.Quotient A J) (Ideal.map (Ideal.Quotien …
    ⊢ Eq ((AlgEquiv.ofRingEquiv ⋯).symm a✝) ((AlgEquiv.ofRingEquiv ⋯) a✝)
  -/
  congr
  /-
    🎉 no goals
  -/


@[simp]
theorem quotQuotEquivComm_comp_quotQuotMkₐ :
    AlgHom.comp (↑(quotQuotEquivCommₐ R I J)) (quotQuotMkₐ R I J) = quotQuotMkₐ R J I :=
  AlgHom.ext <| quotQuotEquivComm_quotQuotMk I J


/-- The **third isomorphism theorem** for algebras. See `quotQuotEquivQuotSupₐ` for version
    that does not assume an inclusion of ideals. -/
def quotQuotEquivQuotOfLEₐ (h : I ≤ J) : ((A ⧸ I) ⧸ J.map (Quotient.mkₐ R I)) ≃ₐ[R] A ⧸ J :=
  AlgEquiv.ofRingEquiv (f := quotQuotEquivQuotOfLE h) fun _ => rfl


@[simp]
theorem quotQuotEquivQuotOfLEₐ_toRingEquiv (h : I ≤ J) :
    (quotQuotEquivQuotOfLEₐ R h : _ ⧸ J.map (Quotient.mkₐ R I) ≃+* _) = quotQuotEquivQuotOfLE h :=
  rfl


@[simp]
-- Porting note: had to add an extra coercion arrow on the right hand side.
theorem coe_quotQuotEquivQuotOfLEₐ (h : I ≤ J) :
    ⇑(quotQuotEquivQuotOfLEₐ R h) = ⇑(quotQuotEquivQuotOfLE h) :=
  rfl


@[simp]
theorem quotQuotEquivQuotOfLEₐ_symm_toRingEquiv (h : I ≤ J) :
    ((quotQuotEquivQuotOfLEₐ R h).symm : _ ≃+* _ ⧸ J.map (Quotient.mkₐ R I)) =
      (quotQuotEquivQuotOfLE h).symm :=
  rfl


@[simp]
-- Porting note: had to add an extra coercion arrow on the right hand side.
theorem coe_quotQuotEquivQuotOfLEₐ_symm (h : I ≤ J) :
    ⇑(quotQuotEquivQuotOfLEₐ R h).symm = ⇑(quotQuotEquivQuotOfLE h).symm :=
  rfl


@[simp]
theorem quotQuotEquivQuotOfLE_comp_quotQuotMkₐ (h : I ≤ J) :
    AlgHom.comp (↑(quotQuotEquivQuotOfLEₐ R h)) (quotQuotMkₐ R I J) = Quotient.mkₐ R J :=
  rfl


@[simp]
theorem quotQuotEquivQuotOfLE_symm_comp_mkₐ (h : I ≤ J) :
    AlgHom.comp (↑(quotQuotEquivQuotOfLEₐ R h).symm) (Quotient.mkₐ R J) = quotQuotMkₐ R I J :=
  rfl


/-- `I ^ n ⧸ I ^ (n + 1)` can be viewed as a quotient module and as ideal of `R ⧸ I ^ (n + 1)`.
This definition gives the `R`-linear equivalence between the two. -/
noncomputable
def powQuotPowSuccLinearEquivMapMkPowSuccPow :
    ((I ^ n : Ideal R) ⧸ (I • ⊤ : Submodule R (I ^ n : Ideal R))) ≃ₗ[R]
    Ideal.map (Ideal.Quotient.mk (I ^ (n + 1))) (I ^ n) := by
  refine { LinearMap.codRestrict
    (Submodule.restrictScalars _ (Ideal.map (Ideal.Quotient.mk (I ^ (n + 1))) (I ^ n)))
    (Submodule.mapQ (I • ⊤) (I ^ (n + 1)) (Submodule.subtype (I ^ n)) ?_) ?_,
    Equiv.ofBijective _ ⟨?_, ?_⟩ with }
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      ⊢ LE.le (HSMul.hSMul I Top.top) (Submodule.comap (Submodule.subtype (HPow.hPow …
    -/
  · intro
    /-
      case refine_1
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      x✝ : Subtype fun x => Membership.mem (HPow.hPow I n) x
      ⊢ Membership.mem (HSMul.hSMul I Top.top) x✝ → Membership.mem (Submodule.comap  …
    -/
    simp [Submodule.mem_smul_top_iff, pow_succ']
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      ⊢ ∀ (c : HasQuotient.Quotient (Subtype fun x => Membership.mem (HPow.hPow I n) …
    -/
  · intro x
    /-
      case refine_2
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      x : HasQuotient.Quotient (Subtype fun x => Membership.mem (HPow.hPow I n) x) ( …
      ⊢ Membership.mem (Submodule.restrictScalars R (Ideal.map (Ideal.Quotient.mk (H …
    -/
    obtain ⟨⟨y, hy⟩, rfl⟩ := Submodule.Quotient.mk_surjective _ x
    /-
      case refine_2.intro.mk
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      y : R
      hy : Membership.mem (HPow.hPow I n) y
      ⊢ Membership.mem (Submodule.restrictScalars R (Ideal.map (Ideal.Quotient.mk (H …
    -/
    simp [Ideal.mem_sup_left hy]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      ⊢ Function.Injective fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (HAd …
    -/
  · intro a b
    /-
      case refine_3
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      a b : HasQuotient.Quotient (Subtype fun x => Membership.mem (HPow.hPow I n) x) …
      ⊢ Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (HAdd.hAdd n 1)) ( …
    -/
    obtain ⟨⟨x, hx⟩, rfl⟩ := Submodule.Quotient.mk_surjective _ a
    /-
      case refine_3.intro.mk
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      b : HasQuotient.Quotient (Subtype fun x => Membership.mem (HPow.hPow I n) x) ( …
      x : R
      hx : Membership.mem (HPow.hPow I n) x
      ⊢ Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (HAdd.hAdd n 1)) ( …
    -/
    obtain ⟨⟨y, hy⟩, rfl⟩ := Submodule.Quotient.mk_surjective _ b
    /-
      case refine_3.intro.mk.intro.mk
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      x : R
      hx : Membership.mem (HPow.hPow I n) x
      y : R
      hy : Membership.mem (HPow.hPow I n) y
      ⊢ Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (HAdd.hAdd n 1)) ( …
    -/
    simp [Ideal.Quotient.eq, Submodule.Quotient.eq, Submodule.mem_smul_top_iff, pow_succ']
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      ⊢ Function.Surjective fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (HA …
    -/
  · intro ⟨x, hx⟩
    /-
      case refine_4
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      x : HasQuotient.Quotient R (HPow.hPow I (HAdd.hAdd n 1))
      hx : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow I (HAdd.hAdd n 1) …
      ⊢ Exists fun a => Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (H …
    -/
    rw [Ideal.mem_map_iff_of_surjective _ Ideal.Quotient.mk_surjective] at hx
    /-
      case refine_4
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      x : HasQuotient.Quotient R (HPow.hPow I (HAdd.hAdd n 1))
      hx✝ : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow I (HAdd.hAdd n 1 …
      hx : Exists fun x_1 => And (Membership.mem (HPow.hPow I n) x_1) (Eq ((Ideal.Qu …
      ⊢ Exists fun a => Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (H …
    -/
    obtain ⟨y, hy, rfl⟩ := hx
    /-
      case refine_4.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      y : R
      hy : Membership.mem (HPow.hPow I n) y
      hx : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow I (HAdd.hAdd n 1) …
      ⊢ Exists fun a => Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (H …
    -/
    refine ⟨Submodule.Quotient.mk ⟨y, hy⟩, ?_⟩
    /-
      case refine_4.intro.intro
      R : Type u_1
      inst✝ : CommRing R
      I : Ideal R
      n : Nat
      y : R
      hy : Membership.mem (HPow.hPow I n) y
      hx : Membership.mem (Ideal.map (Ideal.Quotient.mk (HPow.hPow I (HAdd.hAdd n 1) …
      ⊢ Eq ((fun c => ⟨((HSMul.hSMul I Top.top).mapQ (HPow.hPow I (HAdd.hAdd n 1)) ( …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- `I ^ n ⧸ I ^ (n + 1)` can be viewed as a quotient module and as ideal of `R ⧸ I ^ (n + 1)`.
This definition gives the equivalence between the two, instead of the `R`-linear equivalence,
to bypass typeclass synthesis issues on complex `Module` goals. -/
noncomputable
def powQuotPowSuccEquivMapMkPowSuccPow :
    ((I ^ n : Ideal R) ⧸ (I • ⊤ : Submodule R (I ^ n : Ideal R))) ≃
    Ideal.map (Ideal.Quotient.mk (I ^ (n + 1))) (I ^ n) :=
  powQuotPowSuccLinearEquivMapMkPowSuccPow I n


