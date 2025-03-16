@[to_additive]
theorem ker_eq_bot_of_cancel {f : A →* B} (h : ∀ u v : f.ker →* A, f.comp u = f.comp v → u = v) :
                    /-
                      A : Type u
                      B : Type v
                      inst✝¹ : Group A
                      inst✝ : Group B
                      f : MonoidHom A B
                      h : ∀ (u v : MonoidHom (Subtype fun x => Membership.mem f.ker x) A), Eq (f.com …
                      ⊢ Eq f.ker Bot.bot
                    -/
    f.ker = ⊥ := by simpa using congr_arg range (h f.ker.subtype 1 (by aesop_cat))
                    /-
                      🎉 no goals
                    -/


@[to_additive]
theorem range_eq_top_of_cancel {f : A →* B}
    (h : ∀ u v : B →* B ⧸ f.range, u.comp f = v.comp f → u = v) : f.range = ⊤ := by
  /-
    A : Type u
    B : Type v
    inst✝¹ : CommGroup A
    inst✝ : CommGroup B
    f : MonoidHom A B
    h : ∀ (u v : MonoidHom B (HasQuotient.Quotient B f.range)), Eq (u.comp f) (v.c …
    ⊢ Eq f.range Top.top
  -/
  specialize h 1 (QuotientGroup.mk' _) _
    /-
      A : Type u
      B : Type v
      inst✝¹ : CommGroup A
      inst✝ : CommGroup B
      f : MonoidHom A B
      h : ∀ (u v : MonoidHom B (HasQuotient.Quotient B f.range)), Eq (u.comp f) (v.c …
      ⊢ Eq (MonoidHom.comp 1 f) ((QuotientGroup.mk' f.range).comp f)
    -/
  · ext1 x
    /-
      case h
      A : Type u
      B : Type v
      inst✝¹ : CommGroup A
      inst✝ : CommGroup B
      f : MonoidHom A B
      h : ∀ (u v : MonoidHom B (HasQuotient.Quotient B f.range)), Eq (u.comp f) (v.c …
      x : A
      ⊢ Eq ((MonoidHom.comp 1 f) x) (((QuotientGroup.mk' f.range).comp f) x)
    -/
    simp only [one_apply, coe_comp, coe_mk', Function.comp_apply]
    rw [show (1 : B ⧸ f.range) = (1 : B) from QuotientGroup.mk_one _, QuotientGroup.eq, inv_one,
      one_mul]
    /-
      case h
      A : Type u
      B : Type v
      inst✝¹ : CommGroup A
      inst✝ : CommGroup B
      f : MonoidHom A B
      h : ∀ (u v : MonoidHom B (HasQuotient.Quotient B f.range)), Eq (u.comp f) (v.c …
      x : A
      ⊢ Membership.mem f.range (f x)
    -/
    exact ⟨x, rfl⟩
    /-
      🎉 no goals
    -/
  /-
    A : Type u
    B : Type v
    inst✝¹ : CommGroup A
    inst✝ : CommGroup B
    f : MonoidHom A B
    h : Eq 1 (QuotientGroup.mk' f.range)
    ⊢ Eq f.range Top.top
  -/
  replace h : (QuotientGroup.mk' f.range).ker = (1 : B →* B ⧸ f.range).ker := by rw [h]
  /-
    A : Type u
    B : Type v
    inst✝¹ : CommGroup A
    inst✝ : CommGroup B
    f : MonoidHom A B
    h : Eq (QuotientGroup.mk' f.range).ker (MonoidHom.ker 1)
    ⊢ Eq f.range Top.top
  -/
  rwa [ker_one, QuotientGroup.ker_mk'] at h
  /-
    🎉 no goals
  -/


@[to_additive]
instance (G : Grp) : Group G.α :=
  G.str


@[to_additive]
theorem ker_eq_bot_of_mono [Mono f] : f.ker = ⊥ :=
  MonoidHom.ker_eq_bot_of_cancel fun u _ =>
    (@cancel_mono _ _ _ _ _ f _ (show Grp.of f.ker ⟶ A from u) _).1


@[to_additive]
theorem mono_iff_ker_eq_bot : Mono f ↔ f.ker = ⊥ :=
  ⟨fun _ => ker_eq_bot_of_mono f, fun h =>
    ConcreteCategory.mono_of_injective _ <| (MonoidHom.ker_eq_bot_iff f).1 h⟩


@[to_additive]
theorem mono_iff_injective : Mono f ↔ Function.Injective f :=
  Iff.trans (mono_iff_ker_eq_bot f) <| MonoidHom.ker_eq_bot_iff f


local notation3 "X" => Set.range (· • (f.range : Set B) : B → Set B)


/-- Define `X'` to be the set of all left cosets with an extra point at "infinity".
-/
inductive XWithInfinity
  | fromCoset : X → XWithInfinity
  | infinity : XWithInfinity


local notation "X'" => XWithInfinity f


local notation "∞" => XWithInfinity.infinity


local notation "SX'" => Equiv.Perm X'


instance : SMul B X' where
  smul b x :=
    match x with
    | fromCoset y => fromCoset ⟨b • y, by
          /-
            A B : Grp
            f : Quiver.Hom A B
            b : ↑B
            x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
            y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
            ⊢ Membership.mem (Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f)) (HSMu …
          -/
          rw [← y.2.choose_spec, leftCoset_assoc]
          -- Porting note: should we make `Bundled.α` reducible?
          /-
            A B : Grp
            f : Quiver.Hom A B
            b : ↑B
            x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
            y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
            ⊢ Membership.mem (Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f)) (HSMu …
          -/
          let b' : B := y.2.choose
          /-
            A B : Grp
            f : Quiver.Hom A B
            b : ↑B
            x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
            y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
            b' : ↑B := Exists.choose ⋯
            ⊢ Membership.mem (Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f)) (HSMu …
          -/
          use b * b'⟩
          /-
            🎉 no goals
          -/
    | ∞ => ∞


theorem mul_smul (b b' : B) (x : X') : (b * b') • x = b • b' • x :=
  match x with
  | fromCoset y => by
    /-
      A B : Grp
      f : Quiver.Hom A B
      b b' : ↑B
      x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
      ⊢ Eq (HSMul.hSMul (HMul.hMul b b') (Grp.SurjectiveOfEpiAuxs.XWithInfinity.from …
    -/
    change fromCoset _ = fromCoset _
    /-
      A B : Grp
      f : Quiver.Hom A B
      b b' : ↑B
      x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
      ⊢ Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul (HMul.hMul  …
    -/
    simp only [leftCoset_assoc]
    /-
      🎉 no goals
    -/
  | ∞ => rfl


theorem one_smul (x : X') : (1 : B) • x = x :=
  match x with
  | fromCoset y => by
    /-
      A B : Grp
      f : Quiver.Hom A B
      x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
      ⊢ Eq (HSMul.hSMul 1 (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset y)) (Grp. …
    -/
    change fromCoset _ = fromCoset _
    /-
      A B : Grp
      f : Quiver.Hom A B
      x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
      ⊢ Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul 1 ↑y, ⋯⟩) ( …
    -/
    simp only [one_leftCoset, Subtype.ext_iff_val]
    /-
      🎉 no goals
    -/
  | ∞ => rfl


theorem fromCoset_eq_of_mem_range {b : B} (hb : b ∈ f.range) :
    fromCoset ⟨b • ↑f.range, b, rfl⟩ = fromCoset ⟨f.range, 1, one_leftCoset _⟩ := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Membership.mem (MonoidHom.range f) b
    ⊢ Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul b ↑(MonoidH …
  -/
  congr
  /-
    case e_a.e_val
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Membership.mem (MonoidHom.range f) b
    ⊢ Eq (HSMul.hSMul b ↑(MonoidHom.range f)) ↑(MonoidHom.range f)
  -/
  let b : B.α := b
  /-
    case e_a.e_val
    A B : Grp
    f : Quiver.Hom A B
    b✝ : ↑B
    hb : Membership.mem (MonoidHom.range f) b✝
    b : ↑B := b✝
    ⊢ Eq (HSMul.hSMul b✝ ↑(MonoidHom.range f)) ↑(MonoidHom.range f)
  -/
  change b • (f.range : Set B) = f.range
  /-
    case e_a.e_val
    A B : Grp
    f : Quiver.Hom A B
    b✝ : ↑B
    hb : Membership.mem (MonoidHom.range f) b✝
    b : ↑B := b✝
    ⊢ Eq (HSMul.hSMul b ↑(MonoidHom.range f)) ↑(MonoidHom.range f)
  -/
  nth_rw 2 [show (f.range : Set B.α) = (1 : B) • f.range from (one_leftCoset _).symm]
  /-
    case e_a.e_val
    A B : Grp
    f : Quiver.Hom A B
    b✝ : ↑B
    hb : Membership.mem (MonoidHom.range f) b✝
    b : ↑B := b✝
    ⊢ Eq (HSMul.hSMul b ↑(MonoidHom.range f)) (HSMul.hSMul 1 ↑(MonoidHom.range f))
  -/
  rw [leftCoset_eq_iff, mul_one]
  /-
    case e_a.e_val
    A B : Grp
    f : Quiver.Hom A B
    b✝ : ↑B
    hb : Membership.mem (MonoidHom.range f) b✝
    b : ↑B := b✝
    ⊢ Membership.mem (MonoidHom.range f) (Inv.inv b)
  -/
  exact Subgroup.inv_mem _ hb
  /-
    🎉 no goals
  -/


theorem fromCoset_ne_of_nin_range {b : B} (hb : b ∉ f.range) :
    fromCoset ⟨b • ↑f.range, b, rfl⟩ ≠ fromCoset ⟨f.range, 1, one_leftCoset _⟩ := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    ⊢ Ne (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul b ↑(MonoidH …
  -/
  intro r
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    r : Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul b ↑(Monoi …
    ⊢ False
  -/
  simp only [fromCoset.injEq, Subtype.mk.injEq] at r
  -- Porting note: annoying dance between types CoeSort.coe B, B.α, and B
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    r : Eq (HSMul.hSMul b ↑(MonoidHom.range f)) ↑(MonoidHom.range f)
    ⊢ False
  -/
  let b' : B.α := b
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    r : Eq (HSMul.hSMul b ↑(MonoidHom.range f)) ↑(MonoidHom.range f)
    b' : ↑B := b
    ⊢ False
  -/
  change b' • (f.range : Set B) = f.range at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    b' : ↑B := b
    r : Eq (HSMul.hSMul b' ↑(MonoidHom.range f)) ↑(MonoidHom.range f)
    ⊢ False
  -/
  nth_rw 2 [show (f.range : Set B.α) = (1 : B) • f.range from (one_leftCoset _).symm] at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    b' : ↑B := b
    r : Eq (HSMul.hSMul b' ↑(MonoidHom.range f)) (HSMul.hSMul 1 ↑(MonoidHom.range  …
    ⊢ False
  -/
  rw [leftCoset_eq_iff, mul_one] at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    b' : ↑B := b
    r : Membership.mem (MonoidHom.range f) (Inv.inv b')
    ⊢ False
  -/
  exact hb (inv_inv b ▸ Subgroup.inv_mem _ r)
  /-
    🎉 no goals
  -/


instance : DecidableEq X' :=
  Classical.decEq _


/-- Let `τ` be the permutation on `X'` exchanging `f.range` and the point at infinity.
-/
noncomputable def tau : SX' :=
  Equiv.swap (fromCoset ⟨↑f.range, ⟨1, one_leftCoset _⟩⟩) ∞


local notation "τ" => tau f


theorem τ_apply_infinity : τ ∞ = fromCoset ⟨f.range, 1, one_leftCoset _⟩ :=
  Equiv.swap_apply_right _ _


theorem τ_apply_fromCoset : τ (fromCoset ⟨f.range, 1, one_leftCoset _⟩) = ∞ :=
  Equiv.swap_apply_left _ _


theorem τ_apply_fromCoset' (x : B) (hx : x ∈ f.range) :
    τ (fromCoset ⟨x • ↑f.range, ⟨x, rfl⟩⟩) = ∞ :=
  (fromCoset_eq_of_mem_range _ hx).symm ▸ τ_apply_fromCoset _


theorem τ_symm_apply_fromCoset : Equiv.symm τ (fromCoset ⟨f.range, 1, one_leftCoset _⟩) = ∞ := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    ⊢ Eq ((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)) (Grp.SurjectiveOfEpiAuxs.XW …
  -/
  rw [tau, Equiv.symm_swap, Equiv.swap_apply_left]
  /-
    🎉 no goals
  -/


theorem τ_symm_apply_infinity :
    Equiv.symm τ ∞ = fromCoset ⟨f.range, 1, one_leftCoset _⟩ := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    ⊢ Eq ((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)) Grp.SurjectiveOfEpiAuxs.XWi …
  -/
  rw [tau, Equiv.symm_swap, Equiv.swap_apply_right]
  /-
    🎉 no goals
  -/


/-- Let `g : B ⟶ S(X')` be defined as such that, for any `β : B`, `g(β)` is the function sending
point at infinity to point at infinity and sending coset `y` to `β • y`.
-/
def g : B →* SX' where
  toFun β :=
    { toFun := fun x => β • x
      invFun := fun x => β⁻¹ • x
      left_inv := fun x => by
        /-
          A B : Grp
          f : Quiver.Hom A B
          β : ↑B
          x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
          ⊢ Eq ((fun x => HSMul.hSMul (Inv.inv β) x) ((fun x => HSMul.hSMul β x) x)) x
        -/
        dsimp only
        /-
          A B : Grp
          f : Quiver.Hom A B
          β : ↑B
          x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
          ⊢ Eq (HSMul.hSMul (Inv.inv β) (HSMul.hSMul β x)) x
        -/
        rw [← mul_smul, inv_mul_cancel, one_smul]
        /-
          🎉 no goals
        -/
      right_inv := fun x => by
        /-
          A B : Grp
          f : Quiver.Hom A B
          β : ↑B
          x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
          ⊢ Eq ((fun x => HSMul.hSMul β x) ((fun x => HSMul.hSMul (Inv.inv β) x) x)) x
        -/
        dsimp only
        /-
          A B : Grp
          f : Quiver.Hom A B
          β : ↑B
          x : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
          ⊢ Eq (HSMul.hSMul β (HSMul.hSMul (Inv.inv β) x)) x
        -/
        rw [← mul_smul, mul_inv_cancel, one_smul] }
        /-
          🎉 no goals
        -/
  map_one' := by
    /-
      A B : Grp
      f : Quiver.Hom A B
      ⊢ Eq ((fun β => { toFun := fun x => HSMul.hSMul β x, invFun := fun x => HSMul. …
    -/
    ext
    /-
      case H
      A B : Grp
      f : Quiver.Hom A B
      x✝ : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      ⊢ Eq (((fun β => { toFun := fun x => HSMul.hSMul β x, invFun := fun x => HSMul …
    -/
    simp [one_smul]
    /-
      🎉 no goals
    -/
  map_mul' b1 b2 := by
    /-
      A B : Grp
      f : Quiver.Hom A B
      b1 b2 : ↑B
      ⊢ Eq ({ toFun := fun β => { toFun := fun x => HSMul.hSMul β x, invFun := fun x …
    -/
    ext
    /-
      case H
      A B : Grp
      f : Quiver.Hom A B
      b1 b2 : ↑B
      x✝ : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      ⊢ Eq (({ toFun := fun β => { toFun := fun x => HSMul.hSMul β x, invFun := fun  …
    -/
    simp [mul_smul]
    /-
      🎉 no goals
    -/


local notation "g" => g f


/-- Define `h : B ⟶ S(X')` to be `τ g τ⁻¹`
-/
def h : B →* SX' where
  -- Porting note: mathport removed () from (τ) which are needed
  toFun β := ((τ).symm.trans (g β)).trans τ
  map_one' := by
    /-
      A B : Grp
      f : Quiver.Hom A B
      ⊢ Eq ((fun β => ((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans ((Grp.Surj …
    -/
    ext
    /-
      case H
      A B : Grp
      f : Quiver.Hom A B
      x✝ : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      ⊢ Eq (((fun β => ((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans ((Grp.Sur …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_mul' b1 b2 := by
    /-
      A B : Grp
      f : Quiver.Hom A B
      b1 b2 : ↑B
      ⊢ Eq ({ toFun := fun β => ((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans  …
    -/
    ext
    /-
      case H
      A B : Grp
      f : Quiver.Hom A B
      b1 b2 : ↑B
      x✝ : Grp.SurjectiveOfEpiAuxs.XWithInfinity f
      ⊢ Eq (({ toFun := fun β => ((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans …
    -/
    simp
    /-
      🎉 no goals
    -/


local notation "h" => h f


theorem g_apply_fromCoset (x : B) (y : X) :
    g x (fromCoset y) = fromCoset ⟨x • ↑y,
         /-
           A B : Grp
           f : Quiver.Hom A B
           x : ↑B
           y : ↑(Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f))
           ⊢ Membership.mem (Set.range fun x => HSMul.hSMul x ↑(MonoidHom.range f)) (HSMu …
         -/
      by obtain ⟨z, hz⟩ := y.2; exact ⟨x * z, by simp [← hz, smul_smul]⟩⟩ := rfl
                                /-
                                  🎉 no goals
                                -/


theorem g_apply_infinity (x : B) : (g x) ∞ = ∞ := rfl


theorem h_apply_infinity (x : B) (hx : x ∈ f.range) : (h x) ∞ = ∞ := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) x) Grp.SurjectiveOfEpiAuxs.XWithInfinity. …
  -/
  change ((τ).symm.trans (g x)).trans τ _ = _
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    ⊢ Eq ((((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans ((Grp.SurjectiveOfE …
  -/
  simp only [MonoidHom.coe_mk, Equiv.toFun_as_coe, Equiv.coe_trans, Function.comp_apply]
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    ⊢ Eq ((Grp.SurjectiveOfEpiAuxs.tau f) (((Grp.SurjectiveOfEpiAuxs.g f) x) ((Equ …
  -/
  rw [τ_symm_apply_infinity, g_apply_fromCoset]
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    ⊢ Eq ((Grp.SurjectiveOfEpiAuxs.tau f) (Grp.SurjectiveOfEpiAuxs.XWithInfinity.f …
  -/
  simpa only using τ_apply_fromCoset' f x hx
  /-
    🎉 no goals
  -/


theorem h_apply_fromCoset (x : B) :
    (h x) (fromCoset ⟨f.range, 1, one_leftCoset _⟩) =
      fromCoset ⟨f.range, 1, one_leftCoset _⟩ := by
    /-
      A B : Grp
      f : Quiver.Hom A B
      x : ↑B
      ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) x) (Grp.SurjectiveOfEpiAuxs.XWithInfinity …
    -/
    change ((τ).symm.trans (g x)).trans τ _ = _
    /-
      A B : Grp
      f : Quiver.Hom A B
      x : ↑B
      ⊢ Eq ((((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans ((Grp.SurjectiveOfE …
    -/
    simp [-MonoidHom.coe_range, τ_symm_apply_fromCoset, g_apply_infinity, τ_apply_infinity]
    /-
      🎉 no goals
    -/


theorem h_apply_fromCoset' (x : B) (b : B) (hb : b ∈ f.range) :
    h x (fromCoset ⟨b • f.range, b, rfl⟩) = fromCoset ⟨b • ↑f.range, b, rfl⟩ :=
  (fromCoset_eq_of_mem_range _ hb).symm ▸ h_apply_fromCoset f x


theorem h_apply_fromCoset_nin_range (x : B) (hx : x ∈ f.range) (b : B) (hb : b ∉ f.range) :
    h x (fromCoset ⟨b • f.range, b, rfl⟩) = fromCoset ⟨(x * b) • ↑f.range, x * b, rfl⟩ := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) x) (Grp.SurjectiveOfEpiAuxs.XWithInfinity …
  -/
  change ((τ).symm.trans (g x)).trans τ _ = _
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    ⊢ Eq ((((Equiv.symm (Grp.SurjectiveOfEpiAuxs.tau f)).trans ((Grp.SurjectiveOfE …
  -/
  simp only [tau, MonoidHom.coe_mk, Equiv.toFun_as_coe, Equiv.coe_trans, Function.comp_apply]
  rw [Equiv.symm_swap,
    @Equiv.swap_apply_of_ne_of_ne X' _ (fromCoset ⟨f.range, 1, one_leftCoset _⟩) ∞
      (fromCoset ⟨b • ↑f.range, b, rfl⟩) (fromCoset_ne_of_nin_range _ hb) (by simp)]
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    ⊢ Eq ((Equiv.swap (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨↑(MonoidHo …
  -/
  simp only [g_apply_fromCoset, leftCoset_assoc]
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    ⊢ Eq ((Equiv.swap (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨↑(MonoidHo …
  -/
  refine Equiv.swap_apply_of_ne_of_ne (fromCoset_ne_of_nin_range _ fun r => hb ?_) (by simp)
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    r : Membership.mem (MonoidHom.range f) (HMul.hMul x b)
    ⊢ Membership.mem (MonoidHom.range f) b
  -/
  convert Subgroup.mul_mem _ (Subgroup.inv_mem _ hx) r
  /-
    case h.e'_5
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Membership.mem (MonoidHom.range f) x
    b : ↑B
    hb : Not (Membership.mem (MonoidHom.range f) b)
    r : Membership.mem (MonoidHom.range f) (HMul.hMul x b)
    ⊢ Eq b (HMul.hMul (Inv.inv x) (HMul.hMul x b))
  -/
  rw [← mul_assoc, inv_mul_cancel, one_mul]
  /-
    🎉 no goals
  -/


theorem agree : f.range = { x | h x = g x } := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    ⊢ Eq (↑(MonoidHom.range f)) (setOf fun x => Eq ((Grp.SurjectiveOfEpiAuxs.h f)  …
  -/
  refine Set.ext fun b => ⟨?_, fun hb : h b = g b => by_contradiction fun r => ?_⟩
    /-
      case refine_1
      A B : Grp
      f : Quiver.Hom A B
      b : ↑B
      ⊢ Membership.mem (↑(MonoidHom.range f)) b → Membership.mem (setOf fun x => Eq  …
    -/
  · rintro ⟨a, rfl⟩
    /-
      case refine_1.intro
      A B : Grp
      f : Quiver.Hom A B
      a : ↑A
      ⊢ Membership.mem (setOf fun x => Eq ((Grp.SurjectiveOfEpiAuxs.h f) x) ((Grp.Su …
    -/
    change h (f a) = g (f a)
    /-
      case refine_1.intro
      A B : Grp
      f : Quiver.Hom A B
      a : ↑A
      ⊢ Eq ((Grp.SurjectiveOfEpiAuxs.h f) (f a)) ((Grp.SurjectiveOfEpiAuxs.g f) (f a))
    -/
    ext ⟨⟨_, ⟨y, rfl⟩⟩⟩
      /-
        case refine_1.intro.H.fromCoset.mk.intro
        A B : Grp
        f : Quiver.Hom A B
        a : ↑A
        y : ↑B
        ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) (f a)) (Grp.SurjectiveOfEpiAuxs.XWithInfi …
      -/
    · rw [g_apply_fromCoset]
      /-
        case refine_1.intro.H.fromCoset.mk.intro
        A B : Grp
        f : Quiver.Hom A B
        a : ↑A
        y : ↑B
        ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) (f a)) (Grp.SurjectiveOfEpiAuxs.XWithInfi …
      -/
      by_cases m : y ∈ f.range
        /-
          case pos
          A B : Grp
          f : Quiver.Hom A B
          a : ↑A
          y : ↑B
          m : Membership.mem (MonoidHom.range f) y
          ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) (f a)) (Grp.SurjectiveOfEpiAuxs.XWithInfi …
        -/
      · rw [h_apply_fromCoset' _ _ _ m, fromCoset_eq_of_mem_range _ m]
        /-
          case pos
          A B : Grp
          f : Quiver.Hom A B
          a : ↑A
          y : ↑B
          m : Membership.mem (MonoidHom.range f) y
          ⊢ Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨↑(MonoidHom.range f), ⋯ …
        -/
        change fromCoset _ = fromCoset ⟨f a • (y • _), _⟩
        /-
          case pos
          A B : Grp
          f : Quiver.Hom A B
          a : ↑A
          y : ↑B
          m : Membership.mem (MonoidHom.range f) y
          ⊢ Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨↑(MonoidHom.range f), ⋯ …
        -/
        simp only [← fromCoset_eq_of_mem_range _ (Subgroup.mul_mem _ ⟨a, rfl⟩ m), smul_smul]
        /-
          🎉 no goals
        -/
        /-
          case neg
          A B : Grp
          f : Quiver.Hom A B
          a : ↑A
          y : ↑B
          m : Not (Membership.mem (MonoidHom.range f) y)
          ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) (f a)) (Grp.SurjectiveOfEpiAuxs.XWithInfi …
        -/
      · rw [h_apply_fromCoset_nin_range f (f a) ⟨_, rfl⟩ _ m]
        /-
          case neg
          A B : Grp
          f : Quiver.Hom A B
          a : ↑A
          y : ↑B
          m : Not (Membership.mem (MonoidHom.range f) y)
          ⊢ Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul (HMul.hMul  …
        -/
        simp only [leftCoset_assoc]
        /-
          🎉 no goals
        -/
      /-
        case refine_1.intro.H.infinity
        A B : Grp
        f : Quiver.Hom A B
        a : ↑A
        ⊢ Eq (((Grp.SurjectiveOfEpiAuxs.h f) (f a)) Grp.SurjectiveOfEpiAuxs.XWithInfin …
      -/
    · rw [g_apply_infinity, h_apply_infinity f (f a) ⟨_, rfl⟩]
      /-
        🎉 no goals
      -/
  · have eq1 : (h b) (fromCoset ⟨f.range, 1, one_leftCoset _⟩) =
        fromCoset ⟨f.range, 1, one_leftCoset _⟩ := by
      change ((τ).symm.trans (g b)).trans τ _ = _
      dsimp [tau]
      simp [g_apply_infinity f]
    have eq2 :
      g b (fromCoset ⟨f.range, 1, one_leftCoset _⟩) = fromCoset ⟨b • ↑f.range, b, rfl⟩ := rfl
    /-
      case refine_2
      A B : Grp
      f : Quiver.Hom A B
      b : ↑B
      hb : Eq ((Grp.SurjectiveOfEpiAuxs.h f) b) ((Grp.SurjectiveOfEpiAuxs.g f) b)
      r : Not (Membership.mem (↑(MonoidHom.range f)) b)
      eq1 : Eq (((Grp.SurjectiveOfEpiAuxs.h f) b) (Grp.SurjectiveOfEpiAuxs.XWithInfi …
      eq2 : Eq (((Grp.SurjectiveOfEpiAuxs.g f) b) (Grp.SurjectiveOfEpiAuxs.XWithInfi …
      ⊢ False
    -/
    exact (fromCoset_ne_of_nin_range _ r).symm (by rw [← eq1, ← eq2, DFunLike.congr_fun hb])
    /-
      🎉 no goals
    -/


theorem comp_eq : (f ≫ show B ⟶ Grp.of SX' from g) = f ≫ show B ⟶ Grp.of SX' from h := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    ⊢ Eq (CategoryTheory.CategoryStruct.comp f (letFun (Grp.SurjectiveOfEpiAuxs.g  …
  -/
  ext a
  /-
    case w
    A B : Grp
    f : Quiver.Hom A B
    a : ↑A
    ⊢ Eq ((CategoryTheory.CategoryStruct.comp f (letFun (Grp.SurjectiveOfEpiAuxs.g …
  -/
  change g (f a) = h (f a)
  have : f a ∈ { b | h b = g b } := by
    rw [← agree]
    use a
  /-
    case w
    A B : Grp
    f : Quiver.Hom A B
    a : ↑A
    this : Membership.mem (setOf fun b => Eq ((Grp.SurjectiveOfEpiAuxs.h f) b) ((G …
    ⊢ Eq ((Grp.SurjectiveOfEpiAuxs.g f) (f a)) ((Grp.SurjectiveOfEpiAuxs.h f) (f a))
  -/
  rw [this]
  /-
    🎉 no goals
  -/


theorem g_ne_h (x : B) (hx : x ∉ f.range) : g ≠ h := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Not (Membership.mem (MonoidHom.range f) x)
    ⊢ Ne (Grp.SurjectiveOfEpiAuxs.g f) (Grp.SurjectiveOfEpiAuxs.h f)
  -/
  intro r
  replace r :=
    DFunLike.congr_fun (DFunLike.congr_fun r x) (fromCoset ⟨f.range, ⟨1, one_leftCoset _⟩⟩)
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Not (Membership.mem (MonoidHom.range f) x)
    r : Eq (((Grp.SurjectiveOfEpiAuxs.g f) x) (Grp.SurjectiveOfEpiAuxs.XWithInfini …
    ⊢ False
  -/
  change _ = ((τ).symm.trans (g x)).trans τ _ at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Not (Membership.mem (MonoidHom.range f) x)
    r : Eq (((Grp.SurjectiveOfEpiAuxs.g f) x) (Grp.SurjectiveOfEpiAuxs.XWithInfini …
    ⊢ False
  -/
  rw [g_apply_fromCoset, MonoidHom.coe_mk] at r
  simp only [MonoidHom.coe_range, Subtype.coe_mk, Equiv.symm_swap, Equiv.toFun_as_coe,
    Equiv.coe_trans, Function.comp_apply] at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Not (Membership.mem (MonoidHom.range f) x)
    r : Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul x (Set.ra …
    ⊢ False
  -/
  erw [Equiv.swap_apply_left, g_apply_infinity, Equiv.swap_apply_right] at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    x : ↑B
    hx : Not (Membership.mem (MonoidHom.range f) x)
    r : Eq (Grp.SurjectiveOfEpiAuxs.XWithInfinity.fromCoset ⟨HSMul.hSMul x (Set.ra …
    ⊢ False
  -/
  exact fromCoset_ne_of_nin_range _ hx r
  /-
    🎉 no goals
  -/


theorem surjective_of_epi [Epi f] : Function.Surjective f := by
  /-
    A B : Grp
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Epi f
    ⊢ Function.Surjective ⇑f
  -/
  by_contra r
  /-
    A B : Grp
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Epi f
    r : Not (Function.Surjective ⇑f)
    ⊢ False
  -/
  dsimp [Function.Surjective] at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Epi f
    r : Not (∀ (b : ↑B), Exists fun a => Eq (f a) b)
    ⊢ False
  -/
  push_neg at r
  /-
    A B : Grp
    f : Quiver.Hom A B
    inst✝ : CategoryTheory.Epi f
    r : Exists fun b => ∀ (a : ↑A), Ne (f a) b
    ⊢ False
  -/
  rcases r with ⟨b, hb⟩
  exact
    SurjectiveOfEpiAuxs.g_ne_h f b (fun ⟨c, hc⟩ => hb _ hc)
      ((cancel_epi f).1 (SurjectiveOfEpiAuxs.comp_eq f))


theorem epi_iff_surjective : Epi f ↔ Function.Surjective f :=
  ⟨fun _ => surjective_of_epi f, ConcreteCategory.epi_of_surjective f⟩


theorem epi_iff_range_eq_top : Epi f ↔ f.range = ⊤ :=
  Iff.trans (epi_iff_surjective _) (Subgroup.eq_top_iff' f.range).symm


theorem epi_iff_surjective : Epi f ↔ Function.Surjective f := by
  have i1 : Epi f ↔ Epi (groupAddGroupEquivalence.inverse.map f) := by
    refine ⟨?_, groupAddGroupEquivalence.inverse.epi_of_epi_map⟩
    intro e'
    apply groupAddGroupEquivalence.inverse.map_epi
  /-
    A B : AddGrp
    f : Quiver.Hom A B
    i1 : Iff (CategoryTheory.Epi f) (CategoryTheory.Epi (groupAddGroupEquivalence. …
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑f)
  -/
  rwa [Grp.epi_iff_surjective] at i1
  /-
    🎉 no goals
  -/


theorem epi_iff_range_eq_top : Epi f ↔ f.range = ⊤ :=
  Iff.trans (epi_iff_surjective _) (AddSubgroup.eq_top_iff' f.range).symm


@[to_additive AddGrp.forget_grp_preserves_mono]
instance forget_grp_preserves_mono : (forget Grp).PreservesMonomorphisms where
                      /-
                        A B : Grp
                        f✝ : Quiver.Hom A B
                        X✝ Y✝ : Grp
                        f : Quiver.Hom X✝ Y✝
                        e : CategoryTheory.Mono f
                        ⊢ CategoryTheory.Mono ((CategoryTheory.forget Grp).map f)
                      -/
  preserves f e := by rwa [mono_iff_injective, ← CategoryTheory.mono_iff_injective] at e
                      /-
                        🎉 no goals
                      -/


@[to_additive AddGrp.forget_grp_preserves_epi]
instance forget_grp_preserves_epi : (forget Grp).PreservesEpimorphisms where
                      /-
                        A B : Grp
                        f✝ : Quiver.Hom A B
                        X✝ Y✝ : Grp
                        f : Quiver.Hom X✝ Y✝
                        e : CategoryTheory.Epi f
                        ⊢ CategoryTheory.Epi ((CategoryTheory.forget Grp).map f)
                      -/
  preserves f e := by rwa [epi_iff_surjective, ← CategoryTheory.epi_iff_surjective] at e
                      /-
                        🎉 no goals
                      -/


private instance (A : CommGrp) : CommGroup A.α := A.str

private instance (A : CommGrp) : Group A.α := A.str.toGroup


@[to_additive]
theorem ker_eq_bot_of_mono [Mono f] : f.ker = ⊥ :=
  MonoidHom.ker_eq_bot_of_cancel fun u _ =>
    (@cancel_mono _ _ _ _ _ f _ (show CommGrp.of f.ker ⟶ A from u) _).1


@[to_additive]
theorem range_eq_top_of_epi [Epi f] : f.range = ⊤ :=
  MonoidHom.range_eq_top_of_cancel fun u v h =>
    (@cancel_epi _ _ _ _ _ f _ (show B ⟶ ⟨B ⧸ MonoidHom.range f, inferInstance⟩ from u) v).1 h

-- Porting note: again lack of transparency

@[to_additive]
instance (G : CommGrp) : CommGroup <| (forget CommGrp).obj G :=
  G.str


@[to_additive]
theorem epi_iff_range_eq_top : Epi f ↔ f.range = ⊤ :=
  ⟨fun _ => range_eq_top_of_epi _, fun hf =>
    ConcreteCategory.epi_of_surjective _ <| MonoidHom.range_eq_top.mp hf⟩


@[to_additive]
theorem epi_iff_surjective : Epi f ↔ Function.Surjective f := by
  /-
    A B : CommGrp
    f : Quiver.Hom A B
    ⊢ Iff (CategoryTheory.Epi f) (Function.Surjective ⇑f)
  -/
  rw [epi_iff_range_eq_top, MonoidHom.range_eq_top]
  /-
    🎉 no goals
  -/


@[to_additive AddCommGrp.forget_commGrp_preserves_mono]
instance forget_commGrp_preserves_mono : (forget CommGrp).PreservesMonomorphisms where
                      /-
                        A B : CommGrp
                        f✝ : Quiver.Hom A B
                        X✝ Y✝ : CommGrp
                        f : Quiver.Hom X✝ Y✝
                        e : CategoryTheory.Mono f
                        ⊢ CategoryTheory.Mono ((CategoryTheory.forget CommGrp).map f)
                      -/
  preserves f e := by rwa [mono_iff_injective, ← CategoryTheory.mono_iff_injective] at e
                      /-
                        🎉 no goals
                      -/


@[to_additive AddCommGrp.forget_commGrp_preserves_epi]
instance forget_commGrp_preserves_epi : (forget CommGrp).PreservesEpimorphisms where
                      /-
                        A B : CommGrp
                        f✝ : Quiver.Hom A B
                        X✝ Y✝ : CommGrp
                        f : Quiver.Hom X✝ Y✝
                        e : CategoryTheory.Epi f
                        ⊢ CategoryTheory.Epi ((CategoryTheory.forget CommGrp).map f)
                      -/
  preserves f e := by rwa [epi_iff_surjective, ← CategoryTheory.epi_iff_surjective] at e
                      /-
                        🎉 no goals
                      -/


