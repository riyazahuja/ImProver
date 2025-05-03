/-- The setoid on `R × S` used for the Ore localization. -/
@[to_additive AddOreLocalization.oreEqv "The setoid on `R × S` used for the Ore localization."]
def oreEqv : Setoid (X × S) where
  r rs rs' := ∃ (u : S) (v : R), u • rs'.1 = v • rs.1 ∧ u * rs'.2 = v * rs.2
  iseqv := by
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.140
      inst✝ : MulAction R X
      ⊢ Equivalence fun rs rs' => Exists fun u => Exists fun v => And (Eq (HSMul.hSM …
    -/
    refine ⟨fun _ => ⟨1, 1, by simp⟩, ?_, ?_⟩
      /-
        case refine_1
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        ⊢ ∀ {x y : Prod X (Subtype fun x => Membership.mem S x)}, (Exists fun u => Exi …
      -/
    · rintro ⟨r, s⟩ ⟨r', s'⟩ ⟨u, v, hru, hsu⟩; dsimp only at *
      /-
        case refine_1.mk.mk.intro.intro.intro
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        r : X
        s : Subtype fun x => Membership.mem S x
        r' : X
        s' u : Subtype fun x => Membership.mem S x
        v : R
        hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
        hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u r) (HSMul.hSMul v r') …
      -/
      rcases oreCondition (s : R) s' with ⟨r₂, s₂, h₁⟩
      /-
        case refine_1.mk.mk.intro.intro.intro.mk.mk
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        r : X
        s : Subtype fun x => Membership.mem S x
        r' : X
        s' u : Subtype fun x => Membership.mem S x
        v : R
        hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
        hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        h₁ : Eq (HMul.hMul ↑s₂ ↑s) (HMul.hMul r₂ ↑s')
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u r) (HSMul.hSMul v r') …
      -/
      rcases oreCondition r₂ u with ⟨r₃, s₃, h₂⟩
      have : r₃ * v * s = s₃ * s₂ * s := by
        -- Porting note: the proof used `assoc_rw`
        rw [mul_assoc _ (s₂ : R), h₁, ← mul_assoc, h₂, mul_assoc, ← hsu, ← mul_assoc]
      /-
        case refine_1.mk.mk.intro.intro.intro.mk.mk.mk.mk
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        r : X
        s : Subtype fun x => Membership.mem S x
        r' : X
        s' u : Subtype fun x => Membership.mem S x
        v : R
        hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
        hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        h₁ : Eq (HMul.hMul ↑s₂ ↑s) (HMul.hMul r₂ ↑s')
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₂ : Eq (HMul.hMul (↑s₃) r₂) (HMul.hMul r₃ ↑u)
        this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s) (HMul.hMul (HMul.hMul ↑s₃ ↑s₂) ↑s)
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u r) (HSMul.hSMul v r') …
      -/
      rcases ore_right_cancel (r₃ * v) (s₃ * s₂) s this with ⟨w, hw⟩
      /-
        case refine_1.mk.mk.intro.intro.intro.mk.mk.mk.mk.intro
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        r : X
        s : Subtype fun x => Membership.mem S x
        r' : X
        s' u : Subtype fun x => Membership.mem S x
        v : R
        hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
        hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
        r₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        h₁ : Eq (HMul.hMul ↑s₂ ↑s) (HMul.hMul r₂ ↑s')
        r₃ : R
        s₃ : Subtype fun x => Membership.mem S x
        h₂ : Eq (HMul.hMul (↑s₃) r₂) (HMul.hMul r₃ ↑u)
        this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s) (HMul.hMul (HMul.hMul ↑s₃ ↑s₂) ↑s)
        w : Subtype fun x => Membership.mem S x
        hw : Eq (HMul.hMul (↑w) (HMul.hMul r₃ v)) (HMul.hMul (↑w) (HMul.hMul ↑s₃ ↑s₂))
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u r) (HSMul.hSMul v r') …
      -/
      refine ⟨w * (s₃ * s₂), w * (r₃ * u), ?_, ?_⟩ <;>
        /-
          case refine_1.mk.mk.intro.intro.intro.mk.mk.mk.mk.intro.refine_1
          R : Type u_1
          inst✝² : Monoid R
          S : Submonoid R
          inst✝¹ : OreLocalization.OreSet S
          X : Type ?u.140
          inst✝ : MulAction R X
          r : X
          s : Subtype fun x => Membership.mem S x
          r' : X
          s' u : Subtype fun x => Membership.mem S x
          v : R
          hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
          hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
          r₂ : R
          s₂ : Subtype fun x => Membership.mem S x
          h₁ : Eq (HMul.hMul ↑s₂ ↑s) (HMul.hMul r₂ ↑s')
          r₃ : R
          s₃ : Subtype fun x => Membership.mem S x
          h₂ : Eq (HMul.hMul (↑s₃) r₂) (HMul.hMul r₃ ↑u)
          this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s) (HMul.hMul (HMul.hMul ↑s₃ ↑s₂) ↑s)
          w : Subtype fun x => Membership.mem S x
          hw : Eq (HMul.hMul (↑w) (HMul.hMul r₃ v)) (HMul.hMul (↑w) (HMul.hMul ↑s₃ ↑s₂))
          ⊢ Eq (HSMul.hSMul (HMul.hMul w (HMul.hMul s₃ s₂)) r) (HSMul.hSMul (HMul.hMul ( …
        -/
        simp only [Submonoid.coe_mul, Submonoid.smul_def, ← hw]
        /-
          case refine_1.mk.mk.intro.intro.intro.mk.mk.mk.mk.intro.refine_1
          R : Type u_1
          inst✝² : Monoid R
          S : Submonoid R
          inst✝¹ : OreLocalization.OreSet S
          X : Type ?u.140
          inst✝ : MulAction R X
          r : X
          s : Subtype fun x => Membership.mem S x
          r' : X
          s' u : Subtype fun x => Membership.mem S x
          v : R
          hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
          hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
          r₂ : R
          s₂ : Subtype fun x => Membership.mem S x
          h₁ : Eq (HMul.hMul ↑s₂ ↑s) (HMul.hMul r₂ ↑s')
          r₃ : R
          s₃ : Subtype fun x => Membership.mem S x
          h₂ : Eq (HMul.hMul (↑s₃) r₂) (HMul.hMul r₃ ↑u)
          this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s) (HMul.hMul (HMul.hMul ↑s₃ ↑s₂) ↑s)
          w : Subtype fun x => Membership.mem S x
          hw : Eq (HMul.hMul (↑w) (HMul.hMul r₃ v)) (HMul.hMul (↑w) (HMul.hMul ↑s₃ ↑s₂))
          ⊢ Eq (HSMul.hSMul (HMul.hMul (↑w) (HMul.hMul r₃ v)) r) (HSMul.hSMul (HMul.hMul …
        -/
      · simp only [mul_smul, hru, ← Submonoid.smul_def]
        /-
          🎉 no goals
        -/
        /-
          case refine_1.mk.mk.intro.intro.intro.mk.mk.mk.mk.intro.refine_2
          R : Type u_1
          inst✝² : Monoid R
          S : Submonoid R
          inst✝¹ : OreLocalization.OreSet S
          X : Type ?u.140
          inst✝ : MulAction R X
          r : X
          s : Subtype fun x => Membership.mem S x
          r' : X
          s' u : Subtype fun x => Membership.mem S x
          v : R
          hru : Eq (HSMul.hSMul u r') (HSMul.hSMul v r)
          hsu : Eq (HMul.hMul ↑u ↑s') (HMul.hMul v ↑s)
          r₂ : R
          s₂ : Subtype fun x => Membership.mem S x
          h₁ : Eq (HMul.hMul ↑s₂ ↑s) (HMul.hMul r₂ ↑s')
          r₃ : R
          s₃ : Subtype fun x => Membership.mem S x
          h₂ : Eq (HMul.hMul (↑s₃) r₂) (HMul.hMul r₃ ↑u)
          this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s) (HMul.hMul (HMul.hMul ↑s₃ ↑s₂) ↑s)
          w : Subtype fun x => Membership.mem S x
          hw : Eq (HMul.hMul (↑w) (HMul.hMul r₃ v)) (HMul.hMul (↑w) (HMul.hMul ↑s₃ ↑s₂))
          ⊢ Eq (HMul.hMul (HMul.hMul (↑w) (HMul.hMul r₃ v)) ↑s) (HMul.hMul (HMul.hMul (↑ …
        -/
      · simp only [mul_assoc, hsu]
        /-
          🎉 no goals
        -/
      /-
        case refine_2
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        ⊢ ∀ {x y z : Prod X (Subtype fun x => Membership.mem S x)}, (Exists fun u => E …
      -/
    · rintro ⟨r₁, s₁⟩ ⟨r₂, s₂⟩ ⟨r₃, s₃⟩ ⟨u, v, hur₁, hs₁u⟩ ⟨u', v', hur₂, hs₂u⟩
      /-
        case refine_2.mk.mk.mk.intro.intro.intro.intro.intro.intro
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : X
        s₃ u : Subtype fun x => Membership.mem S x
        v : R
        hur₁ : Eq (HSMul.hSMul u { fst := r₂, snd := s₂ }.1) (HSMul.hSMul v { fst := r …
        hs₁u : Eq (HMul.hMul ↑u ↑{ fst := r₂, snd := s₂ }.2) (HMul.hMul v ↑{ fst := r₁ …
        u' : Subtype fun x => Membership.mem S x
        v' : R
        hur₂ : Eq (HSMul.hSMul u' { fst := r₃, snd := s₃ }.1) (HSMul.hSMul v' { fst := …
        hs₂u : Eq (HMul.hMul ↑u' ↑{ fst := r₃, snd := s₃ }.2) (HMul.hMul v' ↑{ fst :=  …
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u { fst := r₃, snd := s …
      -/
      rcases oreCondition v' u with ⟨r', s', h⟩; dsimp only at *
      /-
        case refine_2.mk.mk.mk.intro.intro.intro.intro.intro.intro.mk.mk
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.140
        inst✝ : MulAction R X
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        s₂ : Subtype fun x => Membership.mem S x
        r₃ : X
        s₃ u : Subtype fun x => Membership.mem S x
        v : R
        hur₁ : Eq (HSMul.hSMul u r₂) (HSMul.hSMul v r₁)
        hs₁u : Eq (HMul.hMul ↑u ↑s₂) (HMul.hMul v ↑s₁)
        u' : Subtype fun x => Membership.mem S x
        v' : R
        hur₂ : Eq (HSMul.hSMul u' r₃) (HSMul.hSMul v' r₂)
        hs₂u : Eq (HMul.hMul ↑u' ↑s₃) (HMul.hMul v' ↑s₂)
        r' : R
        s' : Subtype fun x => Membership.mem S x
        h : Eq (HMul.hMul (↑s') v') (HMul.hMul r' ↑u)
        ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u r₃) (HSMul.hSMul v r₁ …
      -/
      refine ⟨s' * u', r' * v, ?_, ?_⟩ <;>
        /-
          case refine_2.mk.mk.mk.intro.intro.intro.intro.intro.intro.mk.mk.refine_1
          R : Type u_1
          inst✝² : Monoid R
          S : Submonoid R
          inst✝¹ : OreLocalization.OreSet S
          X : Type ?u.140
          inst✝ : MulAction R X
          r₁ : X
          s₁ : Subtype fun x => Membership.mem S x
          r₂ : X
          s₂ : Subtype fun x => Membership.mem S x
          r₃ : X
          s₃ u : Subtype fun x => Membership.mem S x
          v : R
          hur₁ : Eq (HSMul.hSMul u r₂) (HSMul.hSMul v r₁)
          hs₁u : Eq (HMul.hMul ↑u ↑s₂) (HMul.hMul v ↑s₁)
          u' : Subtype fun x => Membership.mem S x
          v' : R
          hur₂ : Eq (HSMul.hSMul u' r₃) (HSMul.hSMul v' r₂)
          hs₂u : Eq (HMul.hMul ↑u' ↑s₃) (HMul.hMul v' ↑s₂)
          r' : R
          s' : Subtype fun x => Membership.mem S x
          h : Eq (HMul.hMul (↑s') v') (HMul.hMul r' ↑u)
          ⊢ Eq (HSMul.hSMul (HMul.hMul s' u') r₃) (HSMul.hSMul (HMul.hMul r' v) r₁)
        -/
        simp only [Submonoid.smul_def, Submonoid.coe_mul, mul_smul, mul_assoc] at *
        /-
          case refine_2.mk.mk.mk.intro.intro.intro.intro.intro.intro.mk.mk.refine_1
          R : Type u_1
          inst✝² : Monoid R
          S : Submonoid R
          inst✝¹ : OreLocalization.OreSet S
          X : Type ?u.140
          inst✝ : MulAction R X
          r₁ : X
          s₁ : Subtype fun x => Membership.mem S x
          r₂ : X
          s₂ : Subtype fun x => Membership.mem S x
          r₃ : X
          s₃ u : Subtype fun x => Membership.mem S x
          v : R
          hur₁ : Eq (HSMul.hSMul (↑u) r₂) (HSMul.hSMul v r₁)
          hs₁u : Eq (HMul.hMul ↑u ↑s₂) (HMul.hMul v ↑s₁)
          u' : Subtype fun x => Membership.mem S x
          v' : R
          hur₂ : Eq (HSMul.hSMul (↑u') r₃) (HSMul.hSMul v' r₂)
          hs₂u : Eq (HMul.hMul ↑u' ↑s₃) (HMul.hMul v' ↑s₂)
          r' : R
          s' : Subtype fun x => Membership.mem S x
          h : Eq (HMul.hMul (↑s') v') (HMul.hMul r' ↑u)
          ⊢ Eq (HSMul.hSMul (↑s') (HSMul.hSMul (↑u') r₃)) (HSMul.hSMul r' (HSMul.hSMul v …
        -/
      · rw [hur₂, smul_smul, h, mul_smul, hur₁]
        /-
          🎉 no goals
        -/
        /-
          case refine_2.mk.mk.mk.intro.intro.intro.intro.intro.intro.mk.mk.refine_2
          R : Type u_1
          inst✝² : Monoid R
          S : Submonoid R
          inst✝¹ : OreLocalization.OreSet S
          X : Type ?u.140
          inst✝ : MulAction R X
          r₁ : X
          s₁ : Subtype fun x => Membership.mem S x
          r₂ : X
          s₂ : Subtype fun x => Membership.mem S x
          r₃ : X
          s₃ u : Subtype fun x => Membership.mem S x
          v : R
          hur₁ : Eq (HSMul.hSMul (↑u) r₂) (HSMul.hSMul v r₁)
          hs₁u : Eq (HMul.hMul ↑u ↑s₂) (HMul.hMul v ↑s₁)
          u' : Subtype fun x => Membership.mem S x
          v' : R
          hur₂ : Eq (HSMul.hSMul (↑u') r₃) (HSMul.hSMul v' r₂)
          hs₂u : Eq (HMul.hMul ↑u' ↑s₃) (HMul.hMul v' ↑s₂)
          r' : R
          s' : Subtype fun x => Membership.mem S x
          h : Eq (HMul.hMul (↑s') v') (HMul.hMul r' ↑u)
          ⊢ Eq (HMul.hMul (↑s') (HMul.hMul ↑u' ↑s₃)) (HMul.hMul r' (HMul.hMul v ↑s₁))
        -/
      · rw [hs₂u, ← mul_assoc, h, mul_assoc, hs₁u]
        /-
          🎉 no goals
        -/


/-- The Ore localization of a monoid and a submonoid fulfilling the Ore condition. -/
@[to_additive AddOreLocalization "The Ore localization of an additive monoid and a submonoid
fulfilling the Ore condition."]
def OreLocalization {R : Type*} [Monoid R] (S : Submonoid R) [OreSet S]
    (X : Type*) [MulAction R X] :=
  Quotient (OreLocalization.oreEqv S X)


@[inherit_doc OreLocalization]
scoped syntax:1075 term noWs atomic("[" term "⁻¹" noWs "]") : term

macro_rules | `($R[$S⁻¹]) => ``(OreLocalization $S $R)


/-- The division in the Ore localization `X[S⁻¹]`, as a fraction of an element of `X` and `S`. -/
@[to_additive "The subtraction in the Ore localization,
as a difference of an element of `X` and `S`."]
def oreDiv (r : X) (s : S) : X[S⁻¹] :=
  Quotient.mk' (r, s)


@[inherit_doc]
infixl:70 " /ₒ " => oreDiv


@[inherit_doc]
infixl:65 " -ₒ " => _root_.AddOreLocalization.oreSub


@[to_additive (attr := elab_as_elim, cases_eliminator, induction_eliminator)]
protected theorem ind {β : X[S⁻¹] → Prop}
    (c : ∀ (r : X) (s : S), β (r /ₒ s)) : ∀ q, β q := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    β : OreLocalization S X → Prop
    c : ∀ (r : X) (s : Subtype fun x => Membership.mem S x), β (OreLocalization.or …
    ⊢ ∀ (q : OreLocalization S X), β q
  -/
  apply Quotient.ind
  /-
    case a
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    β : OreLocalization S X → Prop
    c : ∀ (r : X) (s : Subtype fun x => Membership.mem S x), β (OreLocalization.or …
    ⊢ ∀ (a : Prod X (Subtype fun x => Membership.mem S x)), β (Quotient.mk (OreLoc …
  -/
  rintro ⟨r, s⟩
  /-
    case a.mk
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    β : OreLocalization S X → Prop
    c : ∀ (r : X) (s : Subtype fun x => Membership.mem S x), β (OreLocalization.or …
    r : X
    s : Subtype fun x => Membership.mem S x
    ⊢ β (Quotient.mk (OreLocalization.oreEqv S X) { fst := r, snd := s })
  -/
  exact c r s
  /-
    🎉 no goals
  -/


@[to_additive]
theorem oreDiv_eq_iff {r₁ r₂ : X} {s₁ s₂ : S} :
    r₁ /ₒ s₁ = r₂ /ₒ s₂ ↔ ∃ (u : S) (v : R), u • r₂ = v • r₁ ∧ u * s₂ = v * s₁ :=
  Quotient.eq''


/-- A fraction `r /ₒ s` is equal to its expansion by an arbitrary factor `t` if `t * s ∈ S`. -/
@[to_additive "A difference `r -ₒ s` is equal to its expansion by an
arbitrary translation `t` if `t + s ∈ S`."]
protected theorem expand (r : X) (s : S) (t : R) (hst : t * (s : R) ∈ S) :
    r /ₒ s = t • r /ₒ ⟨t * s, hst⟩ := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    t : R
    hst : Membership.mem S (HMul.hMul t ↑s)
    ⊢ Eq (OreLocalization.oreDiv r s) (OreLocalization.oreDiv (HSMul.hSMul t r) ⟨H …
  -/
  apply Quotient.sound
  /-
    case a
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    t : R
    hst : Membership.mem S (HMul.hMul t ↑s)
    ⊢ HasEquiv.Equiv { fst := r, snd := s } { fst := HSMul.hSMul t r, snd := ⟨HMul …
  -/
  exact ⟨s, s * t, by rw [mul_smul, Submonoid.smul_def], by rw [← mul_assoc]⟩
  /-
    🎉 no goals
  -/


/-- A fraction is equal to its expansion by a factor from `S`. -/
@[to_additive "A difference is equal to its expansion by a summand from `S`."]
protected theorem expand' (r : X) (s s' : S) : r /ₒ s = s' • r /ₒ (s' * s) :=
                                    /-
                                      R : Type u_1
                                      inst✝² : Monoid R
                                      S : Submonoid R
                                      inst✝¹ : OreLocalization.OreSet S
                                      X : Type u_2
                                      inst✝ : MulAction R X
                                      r : X
                                      s s' : Subtype fun x => Membership.mem S x
                                      ⊢ Membership.mem S (HMul.hMul ↑s' ↑s)
                                    -/
  OreLocalization.expand r s s' (by norm_cast; apply SetLike.coe_mem)
                                               /-
                                                 🎉 no goals
                                               -/


/-- Fractions which differ by a factor of the numerator can be proven equal if
those factors expand to equal elements of `R`. -/
@[to_additive "Differences whose minuends differ by a common summand can be proven equal if
those summands expand to equal elements of `R`."]
protected theorem eq_of_num_factor_eq {r r' r₁ r₂ : R} {s t : S} (h : t * r = t * r') :
    r₁ * r * r₂ /ₒ s = r₁ * r' * r₂ /ₒ s := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r r' r₁ r₂ : R
    s t : Subtype fun x => Membership.mem S x
    h : Eq (HMul.hMul (↑t) r) (HMul.hMul (↑t) r')
    ⊢ Eq (OreLocalization.oreDiv (HMul.hMul (HMul.hMul r₁ r) r₂) s) (OreLocalizati …
  -/
  rcases oreCondition r₁ t with ⟨r₁', t', hr₁⟩
  /-
    case mk.mk
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r r' r₁ r₂ : R
    s t : Subtype fun x => Membership.mem S x
    h : Eq (HMul.hMul (↑t) r) (HMul.hMul (↑t) r')
    r₁' : R
    t' : Subtype fun x => Membership.mem S x
    hr₁ : Eq (HMul.hMul (↑t') r₁) (HMul.hMul r₁' ↑t)
    ⊢ Eq (OreLocalization.oreDiv (HMul.hMul (HMul.hMul r₁ r) r₂) s) (OreLocalizati …
  -/
  rw [OreLocalization.expand' _ s t', OreLocalization.expand' _ s t']
  /-
    case mk.mk
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r r' r₁ r₂ : R
    s t : Subtype fun x => Membership.mem S x
    h : Eq (HMul.hMul (↑t) r) (HMul.hMul (↑t) r')
    r₁' : R
    t' : Subtype fun x => Membership.mem S x
    hr₁ : Eq (HMul.hMul (↑t') r₁) (HMul.hMul r₁' ↑t)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul t' (HMul.hMul (HMul.hMul r₁ r) r₂))  …
  -/
  congr 1
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: use `assoc_rw`?
  calc (t' : R) * (r₁ * r * r₂)
      = t' * r₁ * r * r₂ := by simp [← mul_assoc]
    _ = r₁' * t * r * r₂ := by rw [hr₁]
    _ = r₁' * (t * r) * r₂ := by simp [← mul_assoc]
    _ = r₁' * (t * r') * r₂ := by rw [h]
    _ = r₁' * t * r' * r₂ := by simp [← mul_assoc]
    _ = t' * r₁ * r' * r₂ := by rw [hr₁]
    _ = t' * (r₁ * r' * r₂) := by simp [← mul_assoc]


/-- A function or predicate over `X` and `S` can be lifted to `X[S⁻¹]` if it is invariant
under expansion on the left. -/
@[to_additive "A function or predicate over `X` and `S` can be lifted to the localizaton if it is
invariant under expansion on the left."]
def liftExpand {C : Sort*} (P : X → S → C)
    (hP : ∀ (r : X) (t : R) (s : S) (ht : t * s ∈ S), P r s = P (t • r) ⟨t * s, ht⟩) :
    X[S⁻¹] → C :=
  Quotient.lift (fun p : X × S => P p.1 p.2) fun (r₁, s₁) (r₂, s₂) ⟨u, v, hr₂, hs₂⟩ => by
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.19230
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → C
      hP : ∀ (r : X) (t : R) (s : Subtype fun x => Membership.mem S x) (ht : Members …
      x✝² x✝¹ : Prod X (Subtype fun x => Membership.mem S x)
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := r₁, snd := s₁ } { fst := r₂, snd := s₂ }
      u : Subtype fun x => Membership.mem S x
      v : R
      hr₂ : Eq (HSMul.hSMul u { fst := r₂, snd := s₂ }.1) (HSMul.hSMul v { fst := r₁ …
      hs₂ : Eq (HMul.hMul ↑u ↑{ fst := r₂, snd := s₂ }.2) (HMul.hMul v ↑{ fst := r₁, …
      ⊢ Eq ((fun p => P p.1 p.2) { fst := r₁, snd := s₁ }) ((fun p => P p.1 p.2) { f …
    -/
    dsimp at *
    have s₁vS : v * s₁ ∈ S := by
      rw [← hs₂, ← S.coe_mul]
      exact SetLike.coe_mem (u * s₂)
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.19230
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → C
      hP : ∀ (r : X) (t : R) (s : Subtype fun x => Membership.mem S x) (ht : Members …
      x✝² x✝¹ : Prod X (Subtype fun x => Membership.mem S x)
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := r₁, snd := s₁ } { fst := r₂, snd := s₂ }
      u : Subtype fun x => Membership.mem S x
      v : R
      hr₂ : Eq (HSMul.hSMul u r₂) (HSMul.hSMul v r₁)
      hs₂ : Eq (HMul.hMul ↑u ↑s₂) (HMul.hMul v ↑s₁)
      s₁vS : Membership.mem S (HMul.hMul v ↑s₁)
      ⊢ Eq (P r₁ s₁) (P r₂ s₂)
    -/
    replace hs₂ : u * s₂ = ⟨_, s₁vS⟩ := by ext; simp [hs₂]
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.19230
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → C
      hP : ∀ (r : X) (t : R) (s : Subtype fun x => Membership.mem S x) (ht : Members …
      x✝² x✝¹ : Prod X (Subtype fun x => Membership.mem S x)
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := r₁, snd := s₁ } { fst := r₂, snd := s₂ }
      u : Subtype fun x => Membership.mem S x
      v : R
      hr₂ : Eq (HSMul.hSMul u r₂) (HSMul.hSMul v r₁)
      s₁vS : Membership.mem S (HMul.hMul v ↑s₁)
      hs₂ : Eq (HMul.hMul u s₂) ⟨HMul.hMul v ↑s₁, s₁vS⟩
      ⊢ Eq (P r₁ s₁) (P r₂ s₂)
    -/
    rw [hP r₁ v s₁ s₁vS, hP r₂ u s₂ (by norm_cast; rwa [hs₂]), ← hr₂]
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.19230
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → C
      hP : ∀ (r : X) (t : R) (s : Subtype fun x => Membership.mem S x) (ht : Members …
      x✝² x✝¹ : Prod X (Subtype fun x => Membership.mem S x)
      r₁ : X
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      x✝ : HasEquiv.Equiv { fst := r₁, snd := s₁ } { fst := r₂, snd := s₂ }
      u : Subtype fun x => Membership.mem S x
      v : R
      hr₂ : Eq (HSMul.hSMul u r₂) (HSMul.hSMul v r₁)
      s₁vS : Membership.mem S (HMul.hMul v ↑s₁)
      hs₂ : Eq (HMul.hMul u s₂) ⟨HMul.hMul v ↑s₁, s₁vS⟩
      ⊢ Eq (P (HSMul.hSMul u r₂) ⟨HMul.hMul v ↑s₁, s₁vS⟩) (P (HSMul.hSMul (↑u) r₂) ⟨ …
    -/
    simp only [← hs₂]; rfl
                       /-
                         🎉 no goals
                       -/


@[to_additive (attr := simp)]
theorem liftExpand_of {C : Sort*} {P : X → S → C}
    {hP : ∀ (r : X) (t : R) (s : S) (ht : t * s ∈ S), P r s = P (t • r) ⟨t * s, ht⟩} (r : X)
    (s : S) : liftExpand P hP (r /ₒ s) = P r s :=
  rfl


/-- A version of `liftExpand` used to simultaneously lift functions with two arguments
in `X[S⁻¹]`. -/
@[to_additive "A version of `liftExpand` used to simultaneously lift functions with two arguments"]
def lift₂Expand {C : Sort*} (P : X → S → X → S → C)
    (hP :
      ∀ (r₁ : X) (t₁ : R) (s₁ : S) (ht₁ : t₁ * s₁ ∈ S) (r₂ : X) (t₂ : R) (s₂ : S)
        (ht₂ : t₂ * s₂ ∈ S),
        P r₁ s₁ r₂ s₂ = P (t₁ • r₁) ⟨t₁ * s₁, ht₁⟩ (t₂ • r₂) ⟨t₂ * s₂, ht₂⟩) :
    X[S⁻¹] → X[S⁻¹] → C :=
  liftExpand
    (fun r₁ s₁ => liftExpand (P r₁ s₁) fun r₂ t₂ s₂ ht₂ => by
      /-
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.23334
        inst✝ : MulAction R X
        C : Sort u_2
        P : X → (Subtype fun x => Membership.mem S x) → X → (Subtype fun x => Membersh …
        hP : ∀ (r₁ : X) (t₁ : R) (s₁ : Subtype fun x => Membership.mem S x) (ht₁ : Mem …
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        t₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        ht₂ : Membership.mem S (HMul.hMul t₂ ↑s₂)
        ⊢ Eq (P r₁ s₁ r₂ s₂) (P r₁ s₁ (HSMul.hSMul t₂ r₂) ⟨HMul.hMul t₂ ↑s₂, ht₂⟩)
      -/
      have := hP r₁ 1 s₁ (by simp) r₂ t₂ s₂ ht₂
      /-
        R : Type u_1
        inst✝² : Monoid R
        S : Submonoid R
        inst✝¹ : OreLocalization.OreSet S
        X : Type ?u.23334
        inst✝ : MulAction R X
        C : Sort u_2
        P : X → (Subtype fun x => Membership.mem S x) → X → (Subtype fun x => Membersh …
        hP : ∀ (r₁ : X) (t₁ : R) (s₁ : Subtype fun x => Membership.mem S x) (ht₁ : Mem …
        r₁ : X
        s₁ : Subtype fun x => Membership.mem S x
        r₂ : X
        t₂ : R
        s₂ : Subtype fun x => Membership.mem S x
        ht₂ : Membership.mem S (HMul.hMul t₂ ↑s₂)
        this : Eq (P r₁ s₁ r₂ s₂) (P (HSMul.hSMul 1 r₁) ⟨HMul.hMul 1 ↑s₁, ⋯⟩ (HSMul.hS …
        ⊢ Eq (P r₁ s₁ r₂ s₂) (P r₁ s₁ (HSMul.hSMul t₂ r₂) ⟨HMul.hMul t₂ ↑s₂, ht₂⟩)
      -/
      simp [this])
      /-
        🎉 no goals
      -/
    fun r₁ t₁ s₁ ht₁ => by
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.23334
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → X → (Subtype fun x => Membersh …
      hP : ∀ (r₁ : X) (t₁ : R) (s₁ : Subtype fun x => Membership.mem S x) (ht₁ : Mem …
      r₁ : X
      t₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      ht₁ : Membership.mem S (HMul.hMul t₁ ↑s₁)
      ⊢ Eq ((fun r₁ s₁ => OreLocalization.liftExpand (P r₁ s₁) ⋯) r₁ s₁) ((fun r₁ s₁ …
    -/
    ext x; induction' x with r₂ s₂
    /-
      case h.c
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.23334
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → X → (Subtype fun x => Membersh …
      hP : ∀ (r₁ : X) (t₁ : R) (s₁ : Subtype fun x => Membership.mem S x) (ht₁ : Mem …
      r₁ : X
      t₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      ht₁ : Membership.mem S (HMul.hMul t₁ ↑s₁)
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq ((fun r₁ s₁ => OreLocalization.liftExpand (P r₁ s₁) ⋯) r₁ s₁ (OreLocaliza …
    -/
    dsimp only
    /-
      case h.c
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.23334
      inst✝ : MulAction R X
      C : Sort u_2
      P : X → (Subtype fun x => Membership.mem S x) → X → (Subtype fun x => Membersh …
      hP : ∀ (r₁ : X) (t₁ : R) (s₁ : Subtype fun x => Membership.mem S x) (ht₁ : Mem …
      r₁ : X
      t₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      ht₁ : Membership.mem S (HMul.hMul t₁ ↑s₁)
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq (OreLocalization.liftExpand (P r₁ s₁) ⋯ (OreLocalization.oreDiv r₂ s₂)) ( …
    -/
    rw [liftExpand_of, liftExpand_of, hP r₁ t₁ s₁ ht₁ r₂ 1 s₂ (by simp)]; simp
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[to_additive (attr := simp)]
theorem lift₂Expand_of {C : Sort*} {P : X → S → X → S → C}
    {hP :
      ∀ (r₁ : X) (t₁ : R) (s₁ : S) (ht₁ : t₁ * s₁ ∈ S) (r₂ : X) (t₂ : R) (s₂ : S)
        (ht₂ : t₂ * s₂ ∈ S),
        P r₁ s₁ r₂ s₂ = P (t₁ • r₁) ⟨t₁ * s₁, ht₁⟩ (t₂ • r₂) ⟨t₂ * s₂, ht₂⟩}
    (r₁ : X) (s₁ : S) (r₂ : X) (s₂ : S) : lift₂Expand P hP (r₁ /ₒ s₁) (r₂ /ₒ s₂) = P r₁ s₁ r₂ s₂ :=
  rfl


@[to_additive]
private def smul' (r₁ : R) (s₁ : S) (r₂ : X) (s₂ : S) : X[S⁻¹] :=
  oreNum r₁ s₂ • r₂ /ₒ (oreDenom r₁ s₂ * s₁)


@[to_additive]
private theorem smul'_char (r₁ : R) (r₂ : X) (s₁ s₂ : S) (u : S) (v : R) (huv : u * r₁ = v * s₂) :
    OreLocalization.smul' r₁ s₁ r₂ s₂ = v • r₂ /ₒ (u * s₁) := by
  -- Porting note: `assoc_rw` was not ported yet
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    ⊢ Eq (OreLocalization.smul' r₁ s₁ r₂ s₂) (OreLocalization.oreDiv (HSMul.hSMul  …
  -/
  simp only [smul']
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (OreLocalization.oreNum r₁ s₂) r₂) ( …
  -/
  have h₀ := ore_eq r₁ s₂; set v₀ := oreNum r₁ s₂; set u₀ := oreDenom r₁ s₂
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul v₀ r₂) (HMul.hMul u₀ s₁)) (OreLocali …
  -/
  rcases oreCondition (u₀ : R) u with ⟨r₃, s₃, h₃⟩
  have :=
    calc
      r₃ * v * s₂ = r₃ * (u * r₁) := by rw [mul_assoc, ← huv]
      _ = s₃ * (u₀ * r₁) := by rw [← mul_assoc, ← mul_assoc, h₃]
      _ = s₃ * v₀ * s₂ := by rw [mul_assoc, h₀]
  /-
    case mk.mk
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
    this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul v₀ r₂) (HMul.hMul u₀ s₁)) (OreLocali …
  -/
  rcases ore_right_cancel _ _ _ this with ⟨s₄, hs₄⟩
  /-
    case mk.mk.intro
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
    this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
    s₄ : Subtype fun x => Membership.mem S x
    hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul v₀ r₂) (HMul.hMul u₀ s₁)) (OreLocali …
  -/
  symm; rw [oreDiv_eq_iff]
  /-
    case mk.mk.intro
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
    this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
    s₄ : Subtype fun x => Membership.mem S x
    hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
    ⊢ Exists fun u_1 => Exists fun v_1 => And (Eq (HSMul.hSMul u_1 (HSMul.hSMul v₀ …
  -/
  use s₄ * s₃
  /-
    case h
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
    this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
    s₄ : Subtype fun x => Membership.mem S x
    hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
    ⊢ Exists fun v_1 => And (Eq (HSMul.hSMul (HMul.hMul s₄ s₃) (HSMul.hSMul v₀ r₂) …
  -/
  use s₄ * r₃
  /-
    case h
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
    this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
    s₄ : Subtype fun x => Membership.mem S x
    hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
    ⊢ And (Eq (HSMul.hSMul (HMul.hMul s₄ s₃) (HSMul.hSMul v₀ r₂)) (HSMul.hSMul (HM …
  -/
  simp only [Submonoid.coe_mul, Submonoid.smul_def, smul_eq_mul]
  /-
    case h
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ u : Subtype fun x => Membership.mem S x
    v : R
    huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
    v₀ : R := OreLocalization.oreNum r₁ s₂
    u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
    h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
    r₃ : R
    s₃ : Subtype fun x => Membership.mem S x
    h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
    this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
    s₄ : Subtype fun x => Membership.mem S x
    hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
    ⊢ And (Eq (HSMul.hSMul (HMul.hMul ↑s₄ ↑s₃) (HSMul.hSMul v₀ r₂)) (HSMul.hSMul ( …
  -/
  constructor
    /-
      case h.left
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type u_2
      inst✝ : MulAction R X
      r₁ : R
      r₂ : X
      s₁ s₂ u : Subtype fun x => Membership.mem S x
      v : R
      huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
      v₀ : R := OreLocalization.oreNum r₁ s₂
      u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
      h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
      r₃ : R
      s₃ : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
      this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
      s₄ : Subtype fun x => Membership.mem S x
      hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
      ⊢ Eq (HSMul.hSMul (HMul.hMul ↑s₄ ↑s₃) (HSMul.hSMul v₀ r₂)) (HSMul.hSMul (HMul. …
    -/
  · rw [smul_smul, mul_assoc (c := v₀), ← hs₄]
    /-
      case h.left
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type u_2
      inst✝ : MulAction R X
      r₁ : R
      r₂ : X
      s₁ s₂ u : Subtype fun x => Membership.mem S x
      v : R
      huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
      v₀ : R := OreLocalization.oreNum r₁ s₂
      u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
      h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
      r₃ : R
      s₃ : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
      this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
      s₄ : Subtype fun x => Membership.mem S x
      hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
      ⊢ Eq (HSMul.hSMul (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) r₂) (HSMul.hSMul (HMul.hM …
    -/
    simp only [smul_smul, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case h.right
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type u_2
      inst✝ : MulAction R X
      r₁ : R
      r₂ : X
      s₁ s₂ u : Subtype fun x => Membership.mem S x
      v : R
      huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
      v₀ : R := OreLocalization.oreNum r₁ s₂
      u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
      h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
      r₃ : R
      s₃ : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
      this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
      s₄ : Subtype fun x => Membership.mem S x
      hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
      ⊢ Eq (HMul.hMul (HMul.hMul ↑s₄ ↑s₃) (HMul.hMul ↑u₀ ↑s₁)) (HMul.hMul (HMul.hMul …
    -/
  · rw [← mul_assoc (b := (u₀ : R)), mul_assoc (c := (u₀ : R)), h₃]
    /-
      case h.right
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type u_2
      inst✝ : MulAction R X
      r₁ : R
      r₂ : X
      s₁ s₂ u : Subtype fun x => Membership.mem S x
      v : R
      huv : Eq (HMul.hMul (↑u) r₁) (HMul.hMul v ↑s₂)
      v₀ : R := OreLocalization.oreNum r₁ s₂
      u₀ : Subtype fun x => Membership.mem S x := OreLocalization.oreDenom r₁ s₂
      h₀ : Eq (HMul.hMul (↑u₀) r₁) (HMul.hMul v₀ ↑s₂)
      r₃ : R
      s₃ : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃ ↑u₀) (HMul.hMul r₃ ↑u)
      this : Eq (HMul.hMul (HMul.hMul r₃ v) ↑s₂) (HMul.hMul (HMul.hMul (↑s₃) v₀) ↑s₂)
      s₄ : Subtype fun x => Membership.mem S x
      hs₄ : Eq (HMul.hMul (↑s₄) (HMul.hMul r₃ v)) (HMul.hMul (↑s₄) (HMul.hMul (↑s₃)  …
      ⊢ Eq (HMul.hMul (HMul.hMul (↑s₄) (HMul.hMul r₃ ↑u)) ↑s₁) (HMul.hMul (HMul.hMul …
    -/
    simp only [mul_assoc]
    /-
      🎉 no goals
    -/


/-- The multiplication on the Ore localization of monoids. -/
@[to_additive]
private def smul'' (r : R) (s : S) : X[S⁻¹] → X[S⁻¹] :=
  liftExpand (smul' r s) fun r₁ r₂ s' hs => by
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      ⊢ Eq (OreLocalization.smul' r s r₁ s') (OreLocalization.smul' r s (HSMul.hSMul …
    -/
    rcases oreCondition r s' with ⟨r₁', s₁', h₁⟩
    /-
      case mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      ⊢ Eq (OreLocalization.smul' r s r₁ s') (OreLocalization.smul' r s (HSMul.hSMul …
    -/
    rw [smul'_char _ _ _ _ _ _ h₁]
    /-
      case mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' r₁) (HMul.hMul s₁' s)) (OreLocal …
    -/
    rcases oreCondition r ⟨_, hs⟩ with ⟨r₂', s₂', h₂⟩
    /-
      case mk.mk.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' r₁) (HMul.hMul s₁' s)) (OreLocal …
    -/
    rw [smul'_char _ _ _ _ _ _ h₂]
    /-
      case mk.mk.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' r₁) (HMul.hMul s₁' s)) (OreLocal …
    -/
    rcases oreCondition (s₁' : R) (s₂') with ⟨r₃', s₃', h₃⟩
    have : s₃' * r₁' * s' = (r₃' * r₂' * r₂) * s' := by
      rw [mul_assoc, ← h₁, ← mul_assoc, h₃, mul_assoc, h₂]
      simp [mul_assoc]
    /-
      case mk.mk.mk.mk.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃' ↑s₁') (HMul.hMul r₃' ↑s₂')
      this : Eq (HMul.hMul (HMul.hMul (↑s₃') r₁') ↑s') (HMul.hMul (HMul.hMul (HMul.h …
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' r₁) (HMul.hMul s₁' s)) (OreLocal …
    -/
    rcases ore_right_cancel _ _ _ this with ⟨s₄', h₄⟩
    have : (s₄' * r₃') * (s₂' * s) ∈ S := by
      rw [mul_assoc, ← mul_assoc r₃', ← h₃]
      exact (s₄' * (s₃' * s₁' * s)).2
    /-
      case mk.mk.mk.mk.mk.mk.intro
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃' ↑s₁') (HMul.hMul r₃' ↑s₂')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₁') ↑s') (HMul.hMul (HMul.hMul (HMul. …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₁')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₂' ↑s))
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' r₁) (HMul.hMul s₁' s)) (OreLocal …
    -/
    rw [OreLocalization.expand' _ _ (s₄' * s₃'), OreLocalization.expand _ (s₂' * s) _ this]
    /-
      case mk.mk.mk.mk.mk.mk.intro
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃' ↑s₁') (HMul.hMul r₃' ↑s₂')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₁') ↑s') (HMul.hMul (HMul.hMul (HMul. …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₁')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₂' ↑s))
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul s₄' s₃') (HSMul.hSMul r₁' …
    -/
    simp only [Submonoid.smul_def, Submonoid.coe_mul, smul_smul, mul_assoc, h₄]
    /-
      case mk.mk.mk.mk.mk.mk.intro
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃' ↑s₁') (HMul.hMul r₃' ↑s₂')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₁') ↑s') (HMul.hMul (HMul.hMul (HMul. …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₁')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₂' ↑s))
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑s₄') (HMul.hMul r₃' (HM …
    -/
    congr 1
    /-
      case mk.mk.mk.mk.mk.mk.intro.e_s
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃' ↑s₁') (HMul.hMul r₃' ↑s₂')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₁') ↑s') (HMul.hMul (HMul.hMul (HMul. …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₁')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₂' ↑s))
      ⊢ Eq (HMul.hMul s₄' (HMul.hMul s₃' (HMul.hMul s₁' s))) ⟨HMul.hMul (↑s₄') (HMul …
    -/
    ext; simp only [Submonoid.coe_mul, ← mul_assoc]
    /-
      case mk.mk.mk.mk.mk.mk.intro.e_s.a
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.32855
      inst✝ : MulAction R X
      r : R
      s : Subtype fun x => Membership.mem S x
      r₁ : X
      r₂ : R
      s' : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s')
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r) (HMul.hMul r₁' ↑s')
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') r) (HMul.hMul r₂' ↑⟨HMul.hMul r₂ ↑s', hs⟩)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul ↑s₃' ↑s₁') (HMul.hMul r₃' ↑s₂')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₁') ↑s') (HMul.hMul (HMul.hMul (HMul. …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₁')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₂' ↑s))
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul ↑s₄' ↑s₃') ↑s₁') ↑s) (HMul.hMul (HMul.hM …
    -/
    rw [mul_assoc (s₄' : R), h₃, ← mul_assoc]
    /-
      🎉 no goals
    -/


/-- The scalar multiplication on the Ore localization of monoids. -/
@[to_additive (attr := irreducible)
  "the vector addition on the Ore localization of additive monoids."]
protected def smul : R[S⁻¹] → X[S⁻¹] → X[S⁻¹] :=
  liftExpand smul'' fun r₁ r₂ s hs => by
    /-
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      ⊢ Eq (OreLocalization.smul'' r₁ s) (OreLocalization.smul'' (HSMul.hSMul r₂ r₁) …
    -/
    ext x
    /-
      case h
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : OreLocalization S X
      ⊢ Eq (OreLocalization.smul'' r₁ s x) (OreLocalization.smul'' (HSMul.hSMul r₂ r …
    -/
    induction' x with x s₂
    /-
      case h.c
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq (OreLocalization.smul'' r₁ s (OreLocalization.oreDiv x s₂)) (OreLocalizat …
    -/
    show OreLocalization.smul' r₁ s x s₂ = OreLocalization.smul' (r₂ * r₁) ⟨_, hs⟩ x s₂
    /-
      case h.c
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq (OreLocalization.smul' r₁ s x s₂) (OreLocalization.smul' (HMul.hMul r₂ r₁ …
    -/
    rcases oreCondition r₁ s₂ with ⟨r₁', s₁', h₁⟩
    /-
      case h.c.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      ⊢ Eq (OreLocalization.smul' r₁ s x s₂) (OreLocalization.smul' (HMul.hMul r₂ r₁ …
    -/
    rw [smul'_char _ _ _ _ _ _ h₁]
    /-
      case h.c.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' x) (HMul.hMul s₁' s)) (OreLocali …
    -/
    rcases oreCondition (r₂ * r₁) s₂ with ⟨r₂', s₂', h₂⟩
    /-
      case h.c.mk.mk.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' x) (HMul.hMul s₁' s)) (OreLocali …
    -/
    rw [smul'_char _ _ _ _ _ _ h₂]
    /-
      case h.c.mk.mk.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' x) (HMul.hMul s₁' s)) (OreLocali …
    -/
    rcases oreCondition (s₂' * r₂) (s₁') with ⟨r₃', s₃', h₃⟩
    have : s₃' * r₂' * s₂ = r₃' * r₁' * s₂ := by
      rw [mul_assoc, ← h₂, ← mul_assoc _ r₂, ← mul_assoc, h₃, mul_assoc, h₁, mul_assoc]
    /-
      case h.c.mk.mk.mk.mk.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul (↑s₃') (HMul.hMul (↑s₂') r₂)) (HMul.hMul r₃' ↑s₁')
      this : Eq (HMul.hMul (HMul.hMul (↑s₃') r₂') ↑s₂) (HMul.hMul (HMul.hMul r₃' r₁' …
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' x) (HMul.hMul s₁' s)) (OreLocali …
    -/
    rcases ore_right_cancel _ _ _ this with ⟨s₄', h₄⟩
    have : (s₄' * r₃') * (s₁' * s) ∈ S := by
      rw [← mul_assoc, mul_assoc _ r₃', ← h₃, ← mul_assoc, ← mul_assoc, mul_assoc]
      exact mul_mem (s₄' * s₃' * s₂').2 hs
    /-
      case h.c.mk.mk.mk.mk.mk.mk.intro
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul (↑s₃') (HMul.hMul (↑s₂') r₂)) (HMul.hMul r₃' ↑s₁')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₂') ↑s₂) (HMul.hMul (HMul.hMul r₃' r₁ …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₂')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₁' ↑s))
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul r₁' x) (HMul.hMul s₁' s)) (OreLocali …
    -/
    rw [OreLocalization.expand' (r₂' • x) _ (s₄' * s₃'), OreLocalization.expand _ _ _ this]
    /-
      case h.c.mk.mk.mk.mk.mk.mk.intro
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul (↑s₃') (HMul.hMul (↑s₂') r₂)) (HMul.hMul r₃' ↑s₁')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₂') ↑s₂) (HMul.hMul (HMul.hMul r₃' r₁ …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₂')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₁' ↑s))
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑s₄') r₃') (HSMul.hSMul  …
    -/
    simp only [Submonoid.smul_def, Submonoid.coe_mul, smul_smul, mul_assoc, h₄]
    /-
      case h.c.mk.mk.mk.mk.mk.mk.intro
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul (↑s₃') (HMul.hMul (↑s₂') r₂)) (HMul.hMul r₃' ↑s₁')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₂') ↑s₂) (HMul.hMul (HMul.hMul r₃' r₁ …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₂')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₁' ↑s))
      ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul (↑s₄') (HMul.hMul r₃' r₁' …
    -/
    congr 1
    /-
      case h.c.mk.mk.mk.mk.mk.mk.intro.e_s
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul (↑s₃') (HMul.hMul (↑s₂') r₂)) (HMul.hMul r₃' ↑s₁')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₂') ↑s₂) (HMul.hMul (HMul.hMul r₃' r₁ …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₂')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₁' ↑s))
      ⊢ Eq ⟨HMul.hMul (↑s₄') (HMul.hMul r₃' (HMul.hMul ↑s₁' ↑s)), ⋯⟩ (HMul.hMul s₄'  …
    -/
    ext; simp only [Submonoid.coe_mul, ← mul_assoc]
    /-
      case h.c.mk.mk.mk.mk.mk.mk.intro.e_s.a
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type ?u.37382
      inst✝ : MulAction R X
      r₁ r₂ : R
      s : Subtype fun x => Membership.mem S x
      hs : Membership.mem S (HMul.hMul r₂ ↑s)
      x : X
      s₂ : Subtype fun x => Membership.mem S x
      r₁' : R
      s₁' : Subtype fun x => Membership.mem S x
      h₁ : Eq (HMul.hMul (↑s₁') r₁) (HMul.hMul r₁' ↑s₂)
      r₂' : R
      s₂' : Subtype fun x => Membership.mem S x
      h₂ : Eq (HMul.hMul (↑s₂') (HMul.hMul r₂ r₁)) (HMul.hMul r₂' ↑s₂)
      r₃' : R
      s₃' : Subtype fun x => Membership.mem S x
      h₃ : Eq (HMul.hMul (↑s₃') (HMul.hMul (↑s₂') r₂)) (HMul.hMul r₃' ↑s₁')
      this✝ : Eq (HMul.hMul (HMul.hMul (↑s₃') r₂') ↑s₂) (HMul.hMul (HMul.hMul r₃' r₁ …
      s₄' : Subtype fun x => Membership.mem S x
      h₄ : Eq (HMul.hMul (↑s₄') (HMul.hMul (↑s₃') r₂')) (HMul.hMul (↑s₄') (HMul.hMul …
      this : Membership.mem S (HMul.hMul (HMul.hMul (↑s₄') r₃') (HMul.hMul ↑s₁' ↑s))
      ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (↑s₄') r₃') ↑s₁') ↑s) (HMul.hMul (HMul.h …
    -/
    rw [mul_assoc _ r₃', ← h₃, ← mul_assoc, ← mul_assoc]
    /-
      🎉 no goals
    -/


@[to_additive]
instance : SMul R[S⁻¹] X[S⁻¹] :=
  ⟨OreLocalization.smul⟩


@[to_additive]
instance : Mul R[S⁻¹] :=
  ⟨OreLocalization.smul⟩


@[to_additive]
theorem oreDiv_smul_oreDiv {r₁ : R} {r₂ : X} {s₁ s₂ : S} :
    (r₁ /ₒ s₁) • (r₂ /ₒ s₂) = oreNum r₁ s₂ • r₂ /ₒ (oreDenom r₁ s₂ * s₁) := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂ s₂ …
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem oreDiv_mul_oreDiv {r₁ : R} {r₂ : R} {s₁ s₂ : S} :
    (r₁ /ₒ s₁) * (r₂ /ₒ s₂) = oreNum r₁ s₂ * r₂ /ₒ (oreDenom r₁ s₂ * s₁) := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r₁ r₂ : R
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂ s₂)) …
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


/-- A characterization lemma for the scalar multiplication on the Ore localization,
allowing for a choice of Ore numerator and Ore denominator. -/
@[to_additive "A characterization lemma for the vector addition on the Ore localization,
allowing for a choice of Ore minuend and Ore subtrahend."]
theorem oreDiv_smul_char (r₁ : R) (r₂ : X) (s₁ s₂ : S) (r' : R) (s' : S) (huv : s' * r₁ = r' * s₂) :
    (r₁ /ₒ s₁) • (r₂ /ₒ s₂) = r' • r₂ /ₒ (s' * s₁) := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s₁ s₂ : Subtype fun x => Membership.mem S x
    r' : R
    s' : Subtype fun x => Membership.mem S x
    huv : Eq (HMul.hMul (↑s') r₁) (HMul.hMul r' ↑s₂)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂ s₂ …
  -/
  with_unfolding_all exact smul'_char r₁ r₂ s₁ s₂ s' r' huv
  /-
    🎉 no goals
  -/


/-- A characterization lemma for the multiplication on the Ore localization, allowing for a choice
of Ore numerator and Ore denominator. -/
@[to_additive "A characterization lemma for the addition on the Ore localization,
allowing for a choice of Ore minuend and Ore subtrahend."]
theorem oreDiv_mul_char (r₁ r₂ : R) (s₁ s₂ : S) (r' : R) (s' : S) (huv : s' * r₁ = r' * s₂) :
    r₁ /ₒ s₁ * (r₂ /ₒ s₂) = r' * r₂ /ₒ (s' * s₁) := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r₁ r₂ : R
    s₁ s₂ : Subtype fun x => Membership.mem S x
    r' : R
    s' : Subtype fun x => Membership.mem S x
    huv : Eq (HMul.hMul (↑s') r₁) (HMul.hMul r' ↑s₂)
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂ s₂)) …
  -/
  with_unfolding_all exact smul'_char r₁ r₂ s₁ s₂ s' r' huv
  /-
    🎉 no goals
  -/


/-- Another characterization lemma for the scalar multiplication on the Ore localizaion delivering
Ore witnesses and conditions bundled in a sigma type. -/
@[to_additive "Another characterization lemma for the vector addition on the
  Ore localizaion delivering Ore witnesses and conditions bundled in a sigma type."]
def oreDivSMulChar' (r₁ : R) (r₂ : X) (s₁ s₂ : S) :
    Σ'r' : R, Σ's' : S, s' * r₁ = r' * s₂ ∧ (r₁ /ₒ s₁) • (r₂ /ₒ s₂) = r' • r₂ /ₒ (s' * s₁) :=
  ⟨oreNum r₁ s₂, oreDenom r₁ s₂, ore_eq r₁ s₂, oreDiv_smul_oreDiv⟩


/-- Another characterization lemma for the multiplication on the Ore localizaion delivering
Ore witnesses and conditions bundled in a sigma type. -/
@[to_additive "Another characterization lemma for the addition on the Ore localizaion delivering
  Ore witnesses and conditions bundled in a sigma type."]
def oreDivMulChar' (r₁ r₂ : R) (s₁ s₂ : S) :
    Σ'r' : R, Σ's' : S, s' * r₁ = r' * s₂ ∧ r₁ /ₒ s₁ * (r₂ /ₒ s₂) = r' * r₂ /ₒ (s' * s₁) :=
  ⟨oreNum r₁ s₂, oreDenom r₁ s₂, ore_eq r₁ s₂, oreDiv_mul_oreDiv⟩


/-- `1` in the localization, defined as `1 /ₒ 1`. -/
@[to_additive (attr := irreducible) "`0` in the additive localization, defined as `0 -ₒ 0`."]
protected def one : R[S⁻¹] := 1 /ₒ 1


@[to_additive]
instance : One R[S⁻¹] :=
  ⟨OreLocalization.one⟩


@[to_additive]
protected theorem one_def : (1 : R[S⁻¹]) = 1 /ₒ 1 := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    ⊢ Eq 1 (OreLocalization.oreDiv 1 1)
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


@[to_additive]
instance : Inhabited R[S⁻¹] :=
  ⟨1⟩


@[to_additive (attr := simp)]
protected theorem div_eq_one' {r : R} (hr : r ∈ S) : r /ₒ ⟨r, hr⟩ = 1 := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r : R
    hr : Membership.mem S r
    ⊢ Eq (OreLocalization.oreDiv r ⟨r, hr⟩) 1
  -/
  rw [OreLocalization.one_def, oreDiv_eq_iff]
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r : R
    hr : Membership.mem S r
    ⊢ Exists fun u => Exists fun v => And (Eq (HSMul.hSMul u 1) (HSMul.hSMul v r)) …
  -/
  exact ⟨⟨r, hr⟩, 1, by simp, by simp⟩
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem div_eq_one {s : S} : (s : R) /ₒ s = 1 :=
  OreLocalization.div_eq_one' _


@[to_additive]
protected theorem one_smul (x : X[S⁻¹]) : (1 : R[S⁻¹]) • x = x := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul 1 x) x
  -/
  induction' x with r s
  /-
    case c
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul 1 (OreLocalization.oreDiv r s)) (OreLocalization.oreDiv r s)
  -/
  simp [OreLocalization.one_def, oreDiv_smul_char 1 r 1 s 1 s (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem one_mul (x : R[S⁻¹]) : 1 * x = x :=
  OreLocalization.one_smul x


@[to_additive]
protected theorem mul_one (x : R[S⁻¹]) : x * 1 = x := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    x : OreLocalization S R
    ⊢ Eq (HMul.hMul x 1) x
  -/
  induction' x with r s
  /-
    case c
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r s) 1) (OreLocalization.oreDiv r s)
  -/
  simp [OreLocalization.one_def, oreDiv_mul_char r (1 : R) s (1 : S) r 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem mul_smul (x y : R[S⁻¹]) (z : X[S⁻¹]) : (x * y) • z = x • y • z := by
  -- Porting note: `assoc_rw` was not ported yet
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    x y : OreLocalization S R
    z : OreLocalization S X
    ⊢ Eq (HSMul.hSMul (HMul.hMul x y) z) (HSMul.hSMul x (HSMul.hSMul y z))
  -/
  induction' x with r₁ s₁
  /-
    case c
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    y : OreLocalization S R
    z : OreLocalization S X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HMul.hMul (OreLocalization.oreDiv r₁ s₁) y) z) (HSMul.hSMul …
  -/
  induction' y with r₂ s₂
  /-
    case c.c
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    z : OreLocalization S X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HMul.hMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.o …
  -/
  induction' z with r₃ s₃
  /-
    case c.c.c
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HMul.hMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.o …
  -/
  rcases oreDivMulChar' r₁ r₂ s₁ s₂ with ⟨ra, sa, ha, ha'⟩; rw [ha']; clear ha'
  /-
    case c.c.c.mk.mk.intro
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HMul.hMul ra r₂) (HMul.hMul sa s₁)) …
  -/
  rcases oreDivSMulChar' r₂ r₃ s₂ s₃ with ⟨rb, sb, hb, hb'⟩; rw [hb']; clear hb'
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) r₂) (HMul.hMul rb ↑s₃)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HMul.hMul ra r₂) (HMul.hMul sa s₁)) …
  -/
  rcases oreCondition ra sb with ⟨rc, sc, hc⟩
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) r₂) (HMul.hMul rb ↑s₃)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) ra) (HMul.hMul rc ↑sb)
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HMul.hMul ra r₂) (HMul.hMul sa s₁)) …
  -/
  rw [oreDiv_smul_char (ra * r₂) r₃ (sa * s₁) s₃ (rc * rb) sc]; swap
    /-
      case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk
      R : Type u_1
      inst✝² : Monoid R
      S : Submonoid R
      inst✝¹ : OreLocalization.OreSet S
      X : Type u_2
      inst✝ : MulAction R X
      r₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : R
      s₂ : Subtype fun x => Membership.mem S x
      r₃ : X
      s₃ : Subtype fun x => Membership.mem S x
      ra : R
      sa : Subtype fun x => Membership.mem S x
      ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
      rb : R
      sb : Subtype fun x => Membership.mem S x
      hb : Eq (HMul.hMul (↑sb) r₂) (HMul.hMul rb ↑s₃)
      rc : R
      sc : Subtype fun x => Membership.mem S x
      hc : Eq (HMul.hMul (↑sc) ra) (HMul.hMul rc ↑sb)
      ⊢ Eq (HMul.hMul (↑sc) (HMul.hMul ra r₂)) (HMul.hMul (HMul.hMul rc rb) ↑s₃)
    -/
  · rw [← mul_assoc _ ra, hc, mul_assoc, hb, ← mul_assoc]
    /-
      🎉 no goals
    -/
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) r₂) (HMul.hMul rb ↑s₃)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) ra) (HMul.hMul rc ↑sb)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul (HMul.hMul rc rb) r₃) (HMul.hMul sc  …
  -/
  rw [← mul_assoc, mul_smul]
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) r₂) (HMul.hMul rb ↑s₃)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) ra) (HMul.hMul rc ↑sb)
    ⊢ Eq (OreLocalization.oreDiv (HSMul.hSMul rc (HSMul.hSMul rb r₃)) (HMul.hMul ( …
  -/
  symm; apply oreDiv_smul_char
  /-
    case c.c.c.mk.mk.intro.mk.mk.intro.mk.mk.huv
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    s₁ : Subtype fun x => Membership.mem S x
    r₂ : R
    s₂ : Subtype fun x => Membership.mem S x
    r₃ : X
    s₃ : Subtype fun x => Membership.mem S x
    ra : R
    sa : Subtype fun x => Membership.mem S x
    ha : Eq (HMul.hMul (↑sa) r₁) (HMul.hMul ra ↑s₂)
    rb : R
    sb : Subtype fun x => Membership.mem S x
    hb : Eq (HMul.hMul (↑sb) r₂) (HMul.hMul rb ↑s₃)
    rc : R
    sc : Subtype fun x => Membership.mem S x
    hc : Eq (HMul.hMul (↑sc) ra) (HMul.hMul rc ↑sb)
    ⊢ Eq (HMul.hMul (↑(HMul.hMul sc sa)) r₁) (HMul.hMul rc ↑(HMul.hMul sb s₂))
  -/
  rw [Submonoid.coe_mul, Submonoid.coe_mul, ← mul_assoc, ← hc, mul_assoc _ ra, ← ha, mul_assoc]
  /-
    🎉 no goals
  -/


@[to_additive]
protected theorem mul_assoc (x y z : R[S⁻¹]) : x * y * z = x * (y * z) :=
  OreLocalization.mul_smul x y z


/-- `npow` of `OreLocalization` -/
@[to_additive (attr := irreducible) "`nsmul` of `AddOreLocalization`"]
protected def npow : ℕ → R[S⁻¹] → R[S⁻¹] := npowRec


unseal OreLocalization.npow in
@[to_additive]
instance : Monoid R[S⁻¹] where
  one_mul := OreLocalization.one_mul
  mul_one := OreLocalization.mul_one
  mul_assoc := OreLocalization.mul_assoc
  npow := OreLocalization.npow


@[to_additive]
instance instMulActionOreLocalization : MulAction R[S⁻¹] X[S⁻¹] where
  one_smul := OreLocalization.one_smul
  mul_smul := OreLocalization.mul_smul


@[to_additive]
protected theorem mul_inv (s s' : S) : ((s : R) /ₒ s') * ((s' : R) /ₒ s) = 1 := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    s s' : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv (↑s) s') (OreLocalization.oreDiv (↑s') …
  -/
  simp [oreDiv_mul_char (s : R) s' s' s 1 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem one_div_smul {r : X} {s t : S} : ((1 : R) /ₒ t) • (r /ₒ s) = r /ₒ (s * t) := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r : X
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv 1 t) (OreLocalization.oreDiv r s)) ( …
  -/
  simp [oreDiv_smul_char 1 r t s 1 s (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem one_div_mul {r : R} {s t : S} : (1 /ₒ t) * (r /ₒ s) = r /ₒ (s * t) := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r : R
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv 1 t) (OreLocalization.oreDiv r s)) (Or …
  -/
  simp [oreDiv_mul_char 1 r t s 1 s (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem smul_cancel {r : X} {s t : S} : ((s : R) /ₒ t) • (r /ₒ s) = r /ₒ t := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r : X
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (↑s) t) (OreLocalization.oreDiv r s) …
  -/
  simp [oreDiv_smul_char s.1 r t s 1 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem mul_cancel {r : R} {s t : S} : ((s : R) /ₒ t) * (r /ₒ s) = r /ₒ t := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r : R
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv (↑s) t) (OreLocalization.oreDiv r s))  …
  -/
  simp [oreDiv_mul_char s.1 r t s 1 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem smul_cancel' {r₁ : R} {r₂ : X} {s t : S} :
    ((r₁ * s) /ₒ t) • (r₂ /ₒ s) = (r₁ • r₂) /ₒ t := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    r₁ : R
    r₂ : X
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HMul.hMul r₁ ↑s) t) (OreLocalizatio …
  -/
  simp [oreDiv_smul_char (r₁ * s) r₂ t s r₁ 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
protected theorem mul_cancel' {r₁ r₂ : R} {s t : S} :
    ((r₁ * s) /ₒ t) * (r₂ /ₒ s) = (r₁ * r₂) /ₒ t := by
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r₁ r₂ : R
    s t : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv (HMul.hMul r₁ ↑s) t) (OreLocalization. …
  -/
  simp [oreDiv_mul_char (r₁ * s) r₂ t s r₁ 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem smul_div_one {p : R} {r : X} {s : S} : (p /ₒ s) • (r /ₒ 1) = (p • r) /ₒ s := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    X : Type u_2
    inst✝ : MulAction R X
    p : R
    r : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv p s) (OreLocalization.oreDiv r 1)) ( …
  -/
  simp [oreDiv_smul_char p r s 1 p 1 (by simp)]
  /-
    🎉 no goals
  -/


@[to_additive (attr := simp)]
theorem mul_div_one {p r : R} {s : S} : (p /ₒ s) * (r /ₒ 1) = (p * r) /ₒ s := by
  --TODO use coercion r ↦ r /ₒ 1
  /-
    R : Type u_1
    inst✝¹ : Monoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    p r : R
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv p s) (OreLocalization.oreDiv r 1)) (Or …
  -/
  simp [oreDiv_mul_char p r s 1 p 1 (by simp)]
  /-
    🎉 no goals
  -/


/-- The fraction `s /ₒ 1` as a unit in `R[S⁻¹]`, where `s : S`. -/
@[to_additive "The difference `s -ₒ 0` as a an additive unit."]
def numeratorUnit (s : S) : Units R[S⁻¹] where
  val := (s : R) /ₒ 1
  inv := (1 : R) /ₒ s
  val_inv := OreLocalization.mul_inv s 1
  inv_val := OreLocalization.mul_inv 1 s


/-- The multiplicative homomorphism from `R` to `R[S⁻¹]`, mapping `r : R` to the
fraction `r /ₒ 1`. -/
@[to_additive "The additive homomorphism from `R` to `AddOreLocalization R S`,
  mapping `r : R` to the difference `r -ₒ 0`."]
def numeratorHom : R →* R[S⁻¹] where
  toFun r := r /ₒ 1
                 /-
                   R : Type u_1
                   inst✝² : Monoid R
                   S : Submonoid R
                   inst✝¹ : OreLocalization.OreSet S
                   X : Type ?u.69947
                   inst✝ : MulAction R X
                   ⊢ Eq ((fun r => OreLocalization.oreDiv r 1) 1) 1
                 -/
  map_one' := by with_unfolding_all rfl
                 /-
                   🎉 no goals
                 -/
  map_mul' _ _ := mul_div_one.symm


@[to_additive]
theorem numeratorHom_apply {r : R} : numeratorHom r = r /ₒ (1 : S) :=
  rfl


@[to_additive]
theorem numerator_isUnit (s : S) : IsUnit (numeratorHom (s : R) : R[S⁻¹]) :=
  ⟨numeratorUnit s, rfl⟩


/-- The universal lift from a morphism `R →* T`, which maps elements of `S` to units of `T`,
to a morphism `R[S⁻¹] →* T`. -/
@[to_additive "The universal lift from a morphism `R →+ T`, which maps elements of `S` to
  additive-units of `T`, to a morphism `AddOreLocalization R S →+ T`."]
def universalMulHom (hf : ∀ s : S, f s = fS s) : R[S⁻¹] →* T where
  -- Porting note (https://github.com/leanprover-community/mathlib4/issues/12129): additional beta reduction needed
  toFun x :=
    x.liftExpand (fun r s => ((fS s)⁻¹ : Units T) * f r) fun r t s ht => by
      /-
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type ?u.71828
        inst✝¹ : MulAction R X
        T : Type u_2
        inst✝ : Monoid T
        f : MonoidHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        x : OreLocalization S R
        r t : R
        s : Subtype fun x => Membership.mem S x
        ht : Membership.mem S (HMul.hMul t ↑s)
        ⊢ Eq ((fun r s => HMul.hMul (↑(Inv.inv (fS s))) (f r)) r s) ((fun r s => HMul. …
      -/
      simp only [smul_eq_mul]
      have : (fS ⟨t * s, ht⟩ : T) = f t * fS s := by
        simp only [← hf, MonoidHom.map_mul]
      conv_rhs =>
        rw [MonoidHom.map_mul, ← one_mul (f r), ← Units.val_one, ← mul_inv_cancel (fS s)]
        rw [Units.val_mul, mul_assoc, ← mul_assoc _ (fS s : T), ← this, ← mul_assoc]
      /-
        R : Type u_1
        inst✝³ : Monoid R
        S : Submonoid R
        inst✝² : OreLocalization.OreSet S
        X : Type ?u.71828
        inst✝¹ : MulAction R X
        T : Type u_2
        inst✝ : Monoid T
        f : MonoidHom R T
        fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
        hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
        x : OreLocalization S R
        r t : R
        s : Subtype fun x => Membership.mem S x
        ht : Membership.mem S (HMul.hMul t ↑s)
        this : Eq (↑(fS ⟨HMul.hMul t ↑s, ht⟩)) (HMul.hMul (f t) ↑(fS s))
        ⊢ Eq (HMul.hMul (↑(Inv.inv (fS s))) (f r)) (HMul.hMul (HMul.hMul ↑(Inv.inv (fS …
      -/
      simp only [one_mul, Units.inv_mul]
      /-
        🎉 no goals
      -/
                 /-
                   R : Type u_1
                   inst✝³ : Monoid R
                   S : Submonoid R
                   inst✝² : OreLocalization.OreSet S
                   X : Type ?u.71828
                   inst✝¹ : MulAction R X
                   T : Type u_2
                   inst✝ : Monoid T
                   f : MonoidHom R T
                   fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
                   hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
                   ⊢ Eq ((fun x => OreLocalization.liftExpand (fun r s => HMul.hMul (↑(Inv.inv (f …
                 -/
  map_one' := by beta_reduce; rw [OreLocalization.one_def, liftExpand_of]; simp
                                                                           /-
                                                                             🎉 no goals
                                                                           -/
  map_mul' x y := by
    -- Porting note: `simp only []` required, not just for beta reductions
    /-
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type ?u.71828
      inst✝¹ : MulAction R X
      T : Type u_2
      inst✝ : Monoid T
      f : MonoidHom R T
      fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
      x y : OreLocalization S R
      ⊢ Eq ({ toFun := fun x => OreLocalization.liftExpand (fun r s => HMul.hMul (↑( …
    -/
    beta_reduce
    /-
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type ?u.71828
      inst✝¹ : MulAction R X
      T : Type u_2
      inst✝ : Monoid T
      f : MonoidHom R T
      fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
      x y : OreLocalization S R
      ⊢ Eq ({ toFun := fun x => OreLocalization.liftExpand (fun r s => HMul.hMul (↑( …
    -/
    simp only [] -- TODO more!
    /-
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type ?u.71828
      inst✝¹ : MulAction R X
      T : Type u_2
      inst✝ : Monoid T
      f : MonoidHom R T
      fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
      x y : OreLocalization S R
      ⊢ Eq (OreLocalization.liftExpand (fun r s => HMul.hMul (↑(Inv.inv (fS s))) (f  …
    -/
    induction' x with r₁ s₁
    /-
      case c
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type ?u.71828
      inst✝¹ : MulAction R X
      T : Type u_2
      inst✝ : Monoid T
      f : MonoidHom R T
      fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
      y : OreLocalization S R
      r₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      ⊢ Eq (OreLocalization.liftExpand (fun r s => HMul.hMul (↑(Inv.inv (fS s))) (f  …
    -/
    induction' y with r₂ s₂
    /-
      case c.c
      R : Type u_1
      inst✝³ : Monoid R
      S : Submonoid R
      inst✝² : OreLocalization.OreSet S
      X : Type ?u.71828
      inst✝¹ : MulAction R X
      T : Type u_2
      inst✝ : Monoid T
      f : MonoidHom R T
      fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
      hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
      r₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : R
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq (OreLocalization.liftExpand (fun r s => HMul.hMul (↑(Inv.inv (fS s))) (f  …
    -/
    rcases oreDivMulChar' r₁ r₂ s₁ s₂ with ⟨ra, sa, ha, ha'⟩; rw [ha']; clear ha'
    rw [liftExpand_of, liftExpand_of, liftExpand_of, Units.inv_mul_eq_iff_eq_mul, map_mul, map_mul,
      Units.val_mul, mul_assoc, ← mul_assoc (fS s₁ : T), ← mul_assoc (fS s₁ : T), Units.mul_inv,
      one_mul, ← hf, ← mul_assoc, ← map_mul _ _ r₁, ha, map_mul, hf s₂, mul_assoc,
      ← mul_assoc (fS s₂ : T), (fS s₂).mul_inv, one_mul]


@[to_additive]
theorem universalMulHom_apply {r : R} {s : S} :
    universalMulHom f fS hf (r /ₒ s) = ((fS s)⁻¹ : Units T) * f r :=
  rfl


@[to_additive]
theorem universalMulHom_commutes {r : R} : universalMulHom f fS hf (numeratorHom r) = f r := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    T : Type u_2
    inst✝ : Monoid T
    f : MonoidHom R T
    fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
    hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
    r : R
    ⊢ Eq ((OreLocalization.universalMulHom f fS hf) (OreLocalization.numeratorHom  …
  -/
  simp [numeratorHom_apply, universalMulHom_apply]
  /-
    🎉 no goals
  -/


/-- The universal morphism `universalMulHom` is unique. -/
@[to_additive "The universal morphism `universalAddHom` is unique."]
theorem universalMulHom_unique (φ : R[S⁻¹] →* T) (huniv : ∀ r : R, φ (numeratorHom r) = f r) :
    φ = universalMulHom f fS hf := by
  /-
    R : Type u_1
    inst✝² : Monoid R
    S : Submonoid R
    inst✝¹ : OreLocalization.OreSet S
    T : Type u_2
    inst✝ : Monoid T
    f : MonoidHom R T
    fS : MonoidHom (Subtype fun x => Membership.mem S x) (Units T)
    hf : ∀ (s : Subtype fun x => Membership.mem S x), Eq (f ↑s) ↑(fS s)
    φ : MonoidHom (OreLocalization S R) T
    huniv : ∀ (r : R), Eq (φ (OreLocalization.numeratorHom r)) (f r)
    ⊢ Eq φ (OreLocalization.universalMulHom f fS hf)
  -/
  ext x; induction' x with r s
  rw [universalMulHom_apply, ← huniv r, numeratorHom_apply, ← one_mul (φ (r /ₒ s)), ←
    Units.val_one, ← inv_mul_cancel (fS s), Units.val_mul, mul_assoc, ← hf, ← huniv, ← φ.map_mul,
    numeratorHom_apply, OreLocalization.mul_cancel]


/-- Scalar multiplication in a monoid localization. -/
@[to_additive (attr := irreducible) "Vector addition in an additive monoid localization."]
protected def hsmul (c : R) :
    X[S⁻¹] → X[S⁻¹] :=
  liftExpand (fun m s ↦ oreNum (c • 1) s • m /ₒ oreDenom (c • 1) s) (fun r t s ht ↦ by
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹² : Monoid M
      S : Submonoid M
      inst✝¹¹ : OreLocalization.OreSet S
      inst✝¹⁰ : MulAction M X
      inst✝⁹ : SMul R X
      inst✝⁸ : SMul R M
      inst✝⁷ : IsScalarTower R M M
      inst✝⁶ : IsScalarTower R M X
      inst✝⁵ : SMul R' X
      inst✝⁴ : SMul R' M
      inst✝³ : IsScalarTower R' M M
      inst✝² : IsScalarTower R' M X
      inst✝¹ : SMul R R'
      inst✝ : IsScalarTower R R' M
      c : R
      r : X
      t : M
      s : Subtype fun x => Membership.mem S x
      ht : Membership.mem S (HMul.hMul t ↑s)
      ⊢ Eq ((fun m s => OreLocalization.oreDiv (HSMul.hSMul (OreLocalization.oreNum  …
    -/
    dsimp only
    rw [← mul_one (oreDenom (c • 1) s), ← oreDiv_smul_oreDiv, ← mul_one (oreDenom (c • 1) _),
      ← oreDiv_smul_oreDiv, ← OreLocalization.expand])

/- Warning: This gives an diamond on `SMul R[S⁻¹] M[S⁻¹][S⁻¹]`, but we will almost never localize
at the same monoid twice. -/
/- Although the definition does not require `IsScalarTower R M X`,
it does not make sense without it. -/

@[to_additive (attr := nolint unusedArguments)]
instance [SMul R X] [SMul R M] [IsScalarTower R M X] [IsScalarTower R M M] : SMul R (X[S⁻¹]) where
  smul := OreLocalization.hsmul


@[to_additive]
theorem smul_oreDiv (r : R) (x : X) (s : S) :
                                                                    /-
                                                                      R : Type u_1
                                                                      M : Type u_3
                                                                      X : Type u_4
                                                                      inst✝⁶ : Monoid M
                                                                      S : Submonoid M
                                                                      inst✝⁵ : OreLocalization.OreSet S
                                                                      inst✝⁴ : MulAction M X
                                                                      inst✝³ : SMul R X
                                                                      inst✝² : SMul R M
                                                                      inst✝¹ : IsScalarTower R M M
                                                                      inst✝ : IsScalarTower R M X
                                                                      r : R
                                                                      x : X
                                                                      s : Subtype fun x => Membership.mem S x
                                                                      ⊢ Eq (HSMul.hSMul r (OreLocalization.oreDiv x s)) (OreLocalization.oreDiv (HSM …
                                                                    -/
    r • (x /ₒ s) = oreNum (r • 1) s • x /ₒ oreDenom (r • 1) s := by with_unfolding_all rfl
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[to_additive (attr := simp)]
theorem oreDiv_one_smul (r : M) (x : X[S⁻¹]) : (r /ₒ (1 : S)) • x = r • x := by
  /-
    M : Type u_3
    X : Type u_4
    inst✝² : Monoid M
    S : Submonoid M
    inst✝¹ : OreLocalization.OreSet S
    inst✝ : MulAction M X
    r : M
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r 1) x) (HSMul.hSMul r x)
  -/
  induction' x using OreLocalization.ind with r' s
  /-
    case c
    M : Type u_3
    X : Type u_4
    inst✝² : Monoid M
    S : Submonoid M
    inst✝¹ : OreLocalization.OreSet S
    inst✝ : MulAction M X
    r : M
    r' : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv r 1) (OreLocalization.oreDiv r' s))  …
  -/
  rw [smul_oreDiv, oreDiv_smul_oreDiv, mul_one, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_one_smul (r : R) (x : X[S⁻¹]) : (r • 1 : M) • x = r • x := by
  /-
    R : Type u_1
    M : Type u_3
    X : Type u_4
    inst✝⁶ : Monoid M
    S : Submonoid M
    inst✝⁵ : OreLocalization.OreSet S
    inst✝⁴ : MulAction M X
    inst✝³ : SMul R X
    inst✝² : SMul R M
    inst✝¹ : IsScalarTower R M M
    inst✝ : IsScalarTower R M X
    r : R
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul r 1) x) (HSMul.hSMul r x)
  -/
  induction' x using OreLocalization.ind with r' s
  /-
    case c
    R : Type u_1
    M : Type u_3
    X : Type u_4
    inst✝⁶ : Monoid M
    S : Submonoid M
    inst✝⁵ : OreLocalization.OreSet S
    inst✝⁴ : MulAction M X
    inst✝³ : SMul R X
    inst✝² : SMul R M
    inst✝¹ : IsScalarTower R M M
    inst✝ : IsScalarTower R M X
    r : R
    r' : X
    s : Subtype fun x => Membership.mem S x
    ⊢ Eq (HSMul.hSMul (HSMul.hSMul r 1) (OreLocalization.oreDiv r' s)) (HSMul.hSMu …
  -/
  simp only [smul_oreDiv, smul_eq_mul, mul_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem smul_one_oreDiv_one_smul (r : R) (x : X[S⁻¹]) :
    ((r • 1 : M) /ₒ (1 : S)) • x = r • x := by
  /-
    R : Type u_1
    M : Type u_3
    X : Type u_4
    inst✝⁶ : Monoid M
    S : Submonoid M
    inst✝⁵ : OreLocalization.OreSet S
    inst✝⁴ : MulAction M X
    inst✝³ : SMul R X
    inst✝² : SMul R M
    inst✝¹ : IsScalarTower R M M
    inst✝ : IsScalarTower R M X
    r : R
    x : OreLocalization S X
    ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul r 1) 1) x) (HSMul.hSMul …
  -/
  rw [oreDiv_one_smul, smul_one_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
instance : IsScalarTower R R' X[S⁻¹] where
  smul_assoc r m x := by
    rw [← smul_one_oreDiv_one_smul, ← smul_one_oreDiv_one_smul, ← smul_one_oreDiv_one_smul,
      ← mul_smul, mul_div_one]
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹² : Monoid M
      S : Submonoid M
      inst✝¹¹ : OreLocalization.OreSet S
      inst✝¹⁰ : MulAction M X
      inst✝⁹ : SMul R X
      inst✝⁸ : SMul R M
      inst✝⁷ : IsScalarTower R M M
      inst✝⁶ : IsScalarTower R M X
      inst✝⁵ : SMul R' X
      inst✝⁴ : SMul R' M
      inst✝³ : IsScalarTower R' M M
      inst✝² : IsScalarTower R' M X
      inst✝¹ : SMul R R'
      inst✝ : IsScalarTower R R' M
      r : R
      m : R'
      x : OreLocalization S X
      ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HSMul.hSMul (HSMul.hSMul r m) 1) 1) …
    -/
    simp only [smul_eq_mul, mul_one, smul_mul_assoc, smul_assoc, one_mul]
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMulCommClass R R' M] : SMulCommClass R R' X[S⁻¹] where
  smul_comm r m x := by
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹³ : Monoid M
      S : Submonoid M
      inst✝¹² : OreLocalization.OreSet S
      inst✝¹¹ : MulAction M X
      inst✝¹⁰ : SMul R X
      inst✝⁹ : SMul R M
      inst✝⁸ : IsScalarTower R M M
      inst✝⁷ : IsScalarTower R M X
      inst✝⁶ : SMul R' X
      inst✝⁵ : SMul R' M
      inst✝⁴ : IsScalarTower R' M M
      inst✝³ : IsScalarTower R' M X
      inst✝² : SMul R R'
      inst✝¹ : IsScalarTower R R' M
      inst✝ : SMulCommClass R R' M
      r : R
      m : R'
      x : OreLocalization S X
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul m x)) (HSMul.hSMul m (HSMul.hSMul r x))
    -/
    rw [← smul_one_smul m, ← smul_assoc, smul_comm, smul_assoc, smul_one_smul]
    /-
      🎉 no goals
    -/


@[to_additive]
instance : IsScalarTower R M[S⁻¹] X[S⁻¹] where
  smul_assoc r m x := by
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹² : Monoid M
      S : Submonoid M
      inst✝¹¹ : OreLocalization.OreSet S
      inst✝¹⁰ : MulAction M X
      inst✝⁹ : SMul R X
      inst✝⁸ : SMul R M
      inst✝⁷ : IsScalarTower R M M
      inst✝⁶ : IsScalarTower R M X
      inst✝⁵ : SMul R' X
      inst✝⁴ : SMul R' M
      inst✝³ : IsScalarTower R' M M
      inst✝² : IsScalarTower R' M X
      inst✝¹ : SMul R R'
      inst✝ : IsScalarTower R R' M
      r : R
      m : OreLocalization S M
      x : OreLocalization S X
      ⊢ Eq (HSMul.hSMul (HSMul.hSMul r m) x) (HSMul.hSMul r (HSMul.hSMul m x))
    -/
    rw [← smul_one_oreDiv_one_smul, ← smul_one_oreDiv_one_smul, ← mul_smul, smul_eq_mul]
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMulCommClass R M M] : SMulCommClass R M[S⁻¹] X[S⁻¹] where
  smul_comm r x y := by
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹³ : Monoid M
      S : Submonoid M
      inst✝¹² : OreLocalization.OreSet S
      inst✝¹¹ : MulAction M X
      inst✝¹⁰ : SMul R X
      inst✝⁹ : SMul R M
      inst✝⁸ : IsScalarTower R M M
      inst✝⁷ : IsScalarTower R M X
      inst✝⁶ : SMul R' X
      inst✝⁵ : SMul R' M
      inst✝⁴ : IsScalarTower R' M M
      inst✝³ : IsScalarTower R' M X
      inst✝² : SMul R R'
      inst✝¹ : IsScalarTower R R' M
      inst✝ : SMulCommClass R M M
      r : R
      x : OreLocalization S M
      y : OreLocalization S X
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul x y)) (HSMul.hSMul x (HSMul.hSMul r y))
    -/
    induction' x using OreLocalization.ind with r₁ s₁
    /-
      case c
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹³ : Monoid M
      S : Submonoid M
      inst✝¹² : OreLocalization.OreSet S
      inst✝¹¹ : MulAction M X
      inst✝¹⁰ : SMul R X
      inst✝⁹ : SMul R M
      inst✝⁸ : IsScalarTower R M M
      inst✝⁷ : IsScalarTower R M X
      inst✝⁶ : SMul R' X
      inst✝⁵ : SMul R' M
      inst✝⁴ : IsScalarTower R' M M
      inst✝³ : IsScalarTower R' M X
      inst✝² : SMul R R'
      inst✝¹ : IsScalarTower R R' M
      inst✝ : SMulCommClass R M M
      r : R
      y : OreLocalization S X
      r₁ : M
      s₁ : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul r (HSMul.hSMul (OreLocalization.oreDiv r₁ s₁) y)) (HSMul.hSM …
    -/
    induction' y using OreLocalization.ind with r₂ s₂
    rw [← smul_one_oreDiv_one_smul, ← smul_one_oreDiv_one_smul, smul_smul, smul_smul,
      mul_div_one, oreDiv_mul_char _ _ _ _ (r • 1) s₁ (by simp), mul_one]
    /-
      case c.c
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹³ : Monoid M
      S : Submonoid M
      inst✝¹² : OreLocalization.OreSet S
      inst✝¹¹ : MulAction M X
      inst✝¹⁰ : SMul R X
      inst✝⁹ : SMul R M
      inst✝⁸ : IsScalarTower R M M
      inst✝⁷ : IsScalarTower R M X
      inst✝⁶ : SMul R' X
      inst✝⁵ : SMul R' M
      inst✝⁴ : IsScalarTower R' M M
      inst✝³ : IsScalarTower R' M X
      inst✝² : SMul R R'
      inst✝¹ : IsScalarTower R R' M
      inst✝ : SMulCommClass R M M
      r : R
      r₁ : M
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : X
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul (OreLocalization.oreDiv (HMul.hMul (HSMul.hSMul r 1) r₁) s₁) …
    -/
    simp
    /-
      🎉 no goals
    -/


@[to_additive]
instance [SMul Rᵐᵒᵖ M] [SMul Rᵐᵒᵖ X] [IsScalarTower Rᵐᵒᵖ M M] [IsScalarTower Rᵐᵒᵖ M X]
  [IsCentralScalar R M] : IsCentralScalar R X[S⁻¹] where
  op_smul_eq_smul r x := by
    /-
      R : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹⁷ : Monoid M
      S : Submonoid M
      inst✝¹⁶ : OreLocalization.OreSet S
      inst✝¹⁵ : MulAction M X
      inst✝¹⁴ : SMul R X
      inst✝¹³ : SMul R M
      inst✝¹² : IsScalarTower R M M
      inst✝¹¹ : IsScalarTower R M X
      inst✝¹⁰ : SMul R' X
      inst✝⁹ : SMul R' M
      inst✝⁸ : IsScalarTower R' M M
      inst✝⁷ : IsScalarTower R' M X
      inst✝⁶ : SMul R R'
      inst✝⁵ : IsScalarTower R R' M
      inst✝⁴ : SMul (MulOpposite R) M
      inst✝³ : SMul (MulOpposite R) X
      inst✝² : IsScalarTower (MulOpposite R) M M
      inst✝¹ : IsScalarTower (MulOpposite R) M X
      inst✝ : IsCentralScalar R M
      r : R
      x : OreLocalization S X
      ⊢ Eq (HSMul.hSMul (MulOpposite.op r) x) (HSMul.hSMul r x)
    -/
    rw [← smul_one_oreDiv_one_smul, ← smul_one_oreDiv_one_smul, op_smul_eq_smul]
    /-
      🎉 no goals
    -/


@[to_additive]
instance {R} [Monoid R] [MulAction R M] [IsScalarTower R M M]
    [MulAction R X] [IsScalarTower R M X] : MulAction R X[S⁻¹] where
  one_smul := OreLocalization.ind fun x s ↦ by
    /-
      R✝ : Type u_1
      R' : Type u_2
      M : Type u_3
      X : Type u_4
      inst✝¹⁷ : Monoid M
      S : Submonoid M
      inst✝¹⁶ : OreLocalization.OreSet S
      inst✝¹⁵ : MulAction M X
      inst✝¹⁴ : SMul R✝ X
      inst✝¹³ : SMul R✝ M
      inst✝¹² : IsScalarTower R✝ M M
      inst✝¹¹ : IsScalarTower R✝ M X
      inst✝¹⁰ : SMul R' X
      inst✝⁹ : SMul R' M
      inst✝⁸ : IsScalarTower R' M M
      inst✝⁷ : IsScalarTower R' M X
      inst✝⁶ : SMul R✝ R'
      inst✝⁵ : IsScalarTower R✝ R' M
      R : Type ?u.107537
      inst✝⁴ : Monoid R
      inst✝³ : MulAction R M
      inst✝² : IsScalarTower R M M
      inst✝¹ : MulAction R X
      inst✝ : IsScalarTower R M X
      x : X
      s : Subtype fun x => Membership.mem S x
      ⊢ Eq (HSMul.hSMul 1 (OreLocalization.oreDiv x s)) (OreLocalization.oreDiv x s)
    -/
    rw [← smul_one_oreDiv_one_smul, one_smul, ← OreLocalization.one_def, one_smul]
    /-
      🎉 no goals
    -/
                         /-
                           R✝ : Type u_1
                           R' : Type u_2
                           M : Type u_3
                           X : Type u_4
                           inst✝¹⁷ : Monoid M
                           S : Submonoid M
                           inst✝¹⁶ : OreLocalization.OreSet S
                           inst✝¹⁵ : MulAction M X
                           inst✝¹⁴ : SMul R✝ X
                           inst✝¹³ : SMul R✝ M
                           inst✝¹² : IsScalarTower R✝ M M
                           inst✝¹¹ : IsScalarTower R✝ M X
                           inst✝¹⁰ : SMul R' X
                           inst✝⁹ : SMul R' M
                           inst✝⁸ : IsScalarTower R' M M
                           inst✝⁷ : IsScalarTower R' M X
                           inst✝⁶ : SMul R✝ R'
                           inst✝⁵ : IsScalarTower R✝ R' M
                           R : Type ?u.107537
                           inst✝⁴ : Monoid R
                           inst✝³ : MulAction R M
                           inst✝² : IsScalarTower R M M
                           inst✝¹ : MulAction R X
                           inst✝ : IsScalarTower R M X
                           s₁ s₂ : R
                           x : OreLocalization S X
                           ⊢ Eq (HSMul.hSMul (HMul.hMul s₁ s₂) x) (HSMul.hSMul s₁ (HSMul.hSMul s₂ x))
                         -/
  mul_smul s₁ s₂ x := by rw [← smul_eq_mul, smul_assoc]
                         /-
                           🎉 no goals
                         -/


@[to_additive]
theorem smul_oreDiv_one (r : R) (x : X) : r • (x /ₒ (1 : S)) = (r • x) /ₒ (1 : S) := by
  /-
    R : Type u_1
    M : Type u_3
    X : Type u_4
    inst✝⁶ : Monoid M
    S : Submonoid M
    inst✝⁵ : OreLocalization.OreSet S
    inst✝⁴ : MulAction M X
    inst✝³ : SMul R X
    inst✝² : SMul R M
    inst✝¹ : IsScalarTower R M M
    inst✝ : IsScalarTower R M X
    r : R
    x : X
    ⊢ Eq (HSMul.hSMul r (OreLocalization.oreDiv x 1)) (OreLocalization.oreDiv (HSM …
  -/
  rw [← smul_one_oreDiv_one_smul, smul_div_one, smul_assoc, one_smul]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem oreDiv_mul_oreDiv_comm {r₁ r₂ : R} {s₁ s₂ : S} :
    r₁ /ₒ s₁ * (r₂ /ₒ s₂) = r₁ * r₂ /ₒ (s₁ * s₂) := by
  /-
    R : Type u_1
    inst✝¹ : CommMonoid R
    S : Submonoid R
    inst✝ : OreLocalization.OreSet S
    r₁ r₂ : R
    s₁ s₂ : Subtype fun x => Membership.mem S x
    ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂ s₂)) …
  -/
  rw [oreDiv_mul_char r₁ r₂ s₁ s₂ r₁ s₂ (by simp [mul_comm]), mul_comm s₂]
  /-
    🎉 no goals
  -/


@[to_additive]
instance : CommMonoid R[S⁻¹] where
  mul_comm := fun x y => by
    /-
      R : Type u_1
      inst✝¹ : CommMonoid R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      x y : OreLocalization S R
      ⊢ Eq (HMul.hMul x y) (HMul.hMul y x)
    -/
    induction' x with r₁ s₁
    /-
      case c
      R : Type u_1
      inst✝¹ : CommMonoid R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      y : OreLocalization S R
      r₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r₁ s₁) y) (HMul.hMul y (OreLocalizatio …
    -/
    induction' y with r₂ s₂
    /-
      case c.c
      R : Type u_1
      inst✝¹ : CommMonoid R
      S : Submonoid R
      inst✝ : OreLocalization.OreSet S
      r₁ : R
      s₁ : Subtype fun x => Membership.mem S x
      r₂ : R
      s₂ : Subtype fun x => Membership.mem S x
      ⊢ Eq (HMul.hMul (OreLocalization.oreDiv r₁ s₁) (OreLocalization.oreDiv r₂ s₂)) …
    -/
    rw [oreDiv_mul_oreDiv_comm, oreDiv_mul_oreDiv_comm, mul_comm r₁, mul_comm s₁]
    /-
      🎉 no goals
    -/


/-- `0` in the localization, defined as `0 /ₒ 1`. -/
@[irreducible]
protected def zero : X[S⁻¹] := 0 /ₒ 1


instance : Zero X[S⁻¹] :=
  ⟨OreLocalization.zero⟩


protected theorem zero_def : (0 : X[S⁻¹]) = 0 /ₒ 1 := by
  /-
    R : Type u_1
    inst✝³ : Monoid R
    S : Submonoid R
    inst✝² : OreLocalization.OreSet S
    X : Type u_2
    inst✝¹ : Zero X
    inst✝ : MulAction R X
    ⊢ Eq 0 (OreLocalization.oreDiv 0 1)
  -/
  with_unfolding_all rfl
  /-
    🎉 no goals
  -/


