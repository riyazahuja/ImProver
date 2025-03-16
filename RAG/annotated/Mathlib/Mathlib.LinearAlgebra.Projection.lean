theorem ker_id_sub_eq_of_proj {f : E →ₗ[R] p} (hf : ∀ x : p, f x = x) :
    ker (id - p.subtype.comp f) = p := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    ⊢ Eq (LinearMap.ker (HSub.hSub LinearMap.id (p.subtype.comp f))) p
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    x : E
    ⊢ Iff (Membership.mem (LinearMap.ker (HSub.hSub LinearMap.id (p.subtype.comp f …
  -/
  simp only [comp_apply, mem_ker, subtype_apply, sub_apply, id_apply, sub_eq_zero]
  /-
    case h
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    x : E
    ⊢ Iff (Eq x ↑(f x)) (Membership.mem p x)
  -/
  exact ⟨fun h => h.symm ▸ Submodule.coe_mem _, fun hx => by rw [hf ⟨x, hx⟩, Subtype.coe_mk]⟩
  /-
    🎉 no goals
  -/


theorem range_eq_of_proj {f : E →ₗ[R] p} (hf : ∀ x : p, f x = x) : range f = ⊤ :=
  range_eq_top.2 fun x => ⟨x, hf x⟩


theorem isCompl_of_proj {f : E →ₗ[R] p} (hf : ∀ x : p, f x = x) : IsCompl p (ker f) := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    ⊢ IsCompl p (LinearMap.ker f)
  -/
  constructor
    /-
      case disjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      ⊢ Disjoint p (LinearMap.ker f)
    -/
  · rw [disjoint_iff_inf_le]
    /-
      case disjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      ⊢ LE.le (Min.min p (LinearMap.ker f)) Bot.bot
    -/
    rintro x ⟨hpx, hfx⟩
    /-
      case disjoint.intro
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      x : E
      hpx : Membership.mem (↑p) x
      hfx : Membership.mem (↑(LinearMap.ker f)) x
      ⊢ Membership.mem Bot.bot x
    -/
    rw [SetLike.mem_coe, mem_ker, hf ⟨x, hpx⟩, mk_eq_zero] at hfx
    /-
      case disjoint.intro
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      x : E
      hpx : Membership.mem (↑p) x
      hfx : Eq x 0
      ⊢ Membership.mem Bot.bot x
    -/
    simp only [hfx, SetLike.mem_coe, zero_mem]
    /-
      🎉 no goals
    -/
    /-
      case codisjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      ⊢ Codisjoint p (LinearMap.ker f)
    -/
  · rw [codisjoint_iff_le_sup]
    /-
      case codisjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      ⊢ LE.le Top.top (Max.max p (LinearMap.ker f))
    -/
    intro x _
    /-
      case codisjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      x : E
      a✝ : Membership.mem Top.top x
      ⊢ Membership.mem (Max.max p (LinearMap.ker f)) x
    -/
    rw [mem_sup']
    /-
      case codisjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      x : E
      a✝ : Membership.mem Top.top x
      ⊢ Exists fun y => Exists fun z => Eq (HAdd.hAdd ↑y ↑z) x
    -/
    refine ⟨f x, ⟨x - f x, ?_⟩, add_sub_cancel _ _⟩
    /-
      case codisjoint
      R : Type u_1
      inst✝² : Ring R
      E : Type u_2
      inst✝¹ : AddCommGroup E
      inst✝ : Module R E
      p : Submodule R E
      f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
      hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
      x : E
      a✝ : Membership.mem Top.top x
      ⊢ Membership.mem (LinearMap.ker f) (HSub.hSub x ↑(f x))
    -/
    rw [mem_ker, LinearMap.map_sub, hf, sub_self]
    /-
      🎉 no goals
    -/


/-- If `q` is a complement of `p`, then `M/p ≃ q`. -/
def quotientEquivOfIsCompl (h : IsCompl p q) : (E ⧸ p) ≃ₗ[R] q :=
  LinearEquiv.symm <|
    LinearEquiv.ofBijective (p.mkQ.comp q.subtype)
          /-
            R : Type u_1
            inst✝⁹ : Ring R
            E : Type u_2
            inst✝⁸ : AddCommGroup E
            inst✝⁷ : Module R E
            F : Type u_3
            inst✝⁶ : AddCommGroup F
            inst✝⁵ : Module R F
            G : Type u_4
            inst✝⁴ : AddCommGroup G
            inst✝³ : Module R G
            p q : Submodule R E
            S : Type u_5
            inst✝² : Semiring S
            M : Type u_6
            inst✝¹ : AddCommMonoid M
            inst✝ : Module S M
            m : Submodule S M
            h : IsCompl p q
            ⊢ Function.Injective ⇑(p.mkQ.comp q.subtype)
          -/
      ⟨by rw [← ker_eq_bot, ker_comp, ker_mkQ, disjoint_iff_comap_eq_bot.1 h.symm.disjoint], by
          /-
            🎉 no goals
          -/
        /-
          R : Type u_1
          inst✝⁹ : Ring R
          E : Type u_2
          inst✝⁸ : AddCommGroup E
          inst✝⁷ : Module R E
          F : Type u_3
          inst✝⁶ : AddCommGroup F
          inst✝⁵ : Module R F
          G : Type u_4
          inst✝⁴ : AddCommGroup G
          inst✝³ : Module R G
          p q : Submodule R E
          S : Type u_5
          inst✝² : Semiring S
          M : Type u_6
          inst✝¹ : AddCommMonoid M
          inst✝ : Module S M
          m : Submodule S M
          h : IsCompl p q
          ⊢ Function.Surjective ⇑(p.mkQ.comp q.subtype)
        -/
        rw [← range_eq_top, range_comp, range_subtype, map_mkQ_eq_top, h.sup_eq_top]⟩
        /-
          🎉 no goals
        -/


@[simp]
theorem quotientEquivOfIsCompl_symm_apply (h : IsCompl p q) (x : q) :
    -- Porting note: type ascriptions needed on the RHS
    (quotientEquivOfIsCompl p q h).symm x = (Quotient.mk (x : E) : E ⧸ p) := rfl


@[simp]
theorem quotientEquivOfIsCompl_apply_mk_coe (h : IsCompl p q) (x : q) :
    quotientEquivOfIsCompl p q h (Quotient.mk x) = x :=
  (quotientEquivOfIsCompl p q h).apply_symm_apply x


@[simp]
theorem mk_quotientEquivOfIsCompl_apply (h : IsCompl p q) (x : E ⧸ p) :
    (Quotient.mk (quotientEquivOfIsCompl p q h x) : E ⧸ p) = x :=
  (quotientEquivOfIsCompl p q h).symm_apply_apply x


/-- If `q` is a complement of `p`, then `p × q` is isomorphic to `E`. It is the unique
linear map `f : E → p` such that `f x = x` for `x ∈ p` and `f x = 0` for `x ∈ q`. -/
def prodEquivOfIsCompl (h : IsCompl p q) : (p × q) ≃ₗ[R] E := by
  /-
    R : Type u_1
    inst✝⁹ : Ring R
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module R E
    F : Type u_3
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module R F
    G : Type u_4
    inst✝⁴ : AddCommGroup G
    inst✝³ : Module R G
    p q : Submodule R E
    S : Type u_5
    inst✝² : Semiring S
    M : Type u_6
    inst✝¹ : AddCommMonoid M
    inst✝ : Module S M
    m : Submodule S M
    h : IsCompl p q
    ⊢ LinearEquiv (RingHom.id R) (Prod (Subtype fun x => Membership.mem p x) (Subt …
  -/
  apply LinearEquiv.ofBijective (p.subtype.coprod q.subtype)
  /-
    R : Type u_1
    inst✝⁹ : Ring R
    E : Type u_2
    inst✝⁸ : AddCommGroup E
    inst✝⁷ : Module R E
    F : Type u_3
    inst✝⁶ : AddCommGroup F
    inst✝⁵ : Module R F
    G : Type u_4
    inst✝⁴ : AddCommGroup G
    inst✝³ : Module R G
    p q : Submodule R E
    S : Type u_5
    inst✝² : Semiring S
    M : Type u_6
    inst✝¹ : AddCommMonoid M
    inst✝ : Module S M
    m : Submodule S M
    h : IsCompl p q
    ⊢ Function.Bijective ⇑(p.subtype.coprod q.subtype)
  -/
  constructor
    /-
      case left
      R : Type u_1
      inst✝⁹ : Ring R
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module R E
      F : Type u_3
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : Module R F
      G : Type u_4
      inst✝⁴ : AddCommGroup G
      inst✝³ : Module R G
      p q : Submodule R E
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      m : Submodule S M
      h : IsCompl p q
      ⊢ Function.Injective ⇑(p.subtype.coprod q.subtype)
    -/
  · rw [← ker_eq_bot, ker_coprod_of_disjoint_range, ker_subtype, ker_subtype, prod_bot]
    /-
      case left.hd
      R : Type u_1
      inst✝⁹ : Ring R
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module R E
      F : Type u_3
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : Module R F
      G : Type u_4
      inst✝⁴ : AddCommGroup G
      inst✝³ : Module R G
      p q : Submodule R E
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      m : Submodule S M
      h : IsCompl p q
      ⊢ Disjoint (LinearMap.range p.subtype) (LinearMap.range q.subtype)
    -/
    rw [range_subtype, range_subtype]
    /-
      case left.hd
      R : Type u_1
      inst✝⁹ : Ring R
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module R E
      F : Type u_3
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : Module R F
      G : Type u_4
      inst✝⁴ : AddCommGroup G
      inst✝³ : Module R G
      p q : Submodule R E
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      m : Submodule S M
      h : IsCompl p q
      ⊢ Disjoint p q
    -/
    exact h.1
    /-
      🎉 no goals
    -/
    /-
      case right
      R : Type u_1
      inst✝⁹ : Ring R
      E : Type u_2
      inst✝⁸ : AddCommGroup E
      inst✝⁷ : Module R E
      F : Type u_3
      inst✝⁶ : AddCommGroup F
      inst✝⁵ : Module R F
      G : Type u_4
      inst✝⁴ : AddCommGroup G
      inst✝³ : Module R G
      p q : Submodule R E
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      m : Submodule S M
      h : IsCompl p q
      ⊢ Function.Surjective ⇑(p.subtype.coprod q.subtype)
    -/
  · rw [← range_eq_top, ← sup_eq_range, h.sup_eq_top]
    /-
      🎉 no goals
    -/


@[simp]
theorem coe_prodEquivOfIsCompl (h : IsCompl p q) :
    (prodEquivOfIsCompl p q h : p × q →ₗ[R] E) = p.subtype.coprod q.subtype := rfl


@[simp]
theorem coe_prodEquivOfIsCompl' (h : IsCompl p q) (x : p × q) :
    prodEquivOfIsCompl p q h x = x.1 + x.2 := rfl


@[simp]
theorem prodEquivOfIsCompl_symm_apply_left (h : IsCompl p q) (x : p) :
    (prodEquivOfIsCompl p q h).symm x = (x, 0) :=
                                                   /-
                                                     R : Type u_1
                                                     inst✝² : Ring R
                                                     E : Type u_2
                                                     inst✝¹ : AddCommGroup E
                                                     inst✝ : Module R E
                                                     p q : Submodule R E
                                                     h : IsCompl p q
                                                     x : Subtype fun x => Membership.mem p x
                                                     ⊢ Eq (↑x) ((p.prodEquivOfIsCompl q h) { fst := x, snd := 0 })
                                                   -/
  (prodEquivOfIsCompl p q h).symm_apply_eq.2 <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem prodEquivOfIsCompl_symm_apply_right (h : IsCompl p q) (x : q) :
    (prodEquivOfIsCompl p q h).symm x = (0, x) :=
                                                   /-
                                                     R : Type u_1
                                                     inst✝² : Ring R
                                                     E : Type u_2
                                                     inst✝¹ : AddCommGroup E
                                                     inst✝ : Module R E
                                                     p q : Submodule R E
                                                     h : IsCompl p q
                                                     x : Subtype fun x => Membership.mem q x
                                                     ⊢ Eq (↑x) ((p.prodEquivOfIsCompl q h) { fst := 0, snd := x })
                                                   -/
  (prodEquivOfIsCompl p q h).symm_apply_eq.2 <| by simp
                                                   /-
                                                     🎉 no goals
                                                   -/


@[simp]
theorem prodEquivOfIsCompl_symm_apply_fst_eq_zero (h : IsCompl p q) {x : E} :
    ((prodEquivOfIsCompl p q h).symm x).1 = 0 ↔ x ∈ q := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p q : Submodule R E
    h : IsCompl p q
    x : E
    ⊢ Iff (Eq ((p.prodEquivOfIsCompl q h).symm x).1 0) (Membership.mem q x)
  -/
  conv_rhs => rw [← (prodEquivOfIsCompl p q h).apply_symm_apply x]
  rw [coe_prodEquivOfIsCompl', Submodule.add_mem_iff_left _ (Submodule.coe_mem _),
    mem_right_iff_eq_zero_of_disjoint h.disjoint]


@[simp]
theorem prodEquivOfIsCompl_symm_apply_snd_eq_zero (h : IsCompl p q) {x : E} :
    ((prodEquivOfIsCompl p q h).symm x).2 = 0 ↔ x ∈ p := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p q : Submodule R E
    h : IsCompl p q
    x : E
    ⊢ Iff (Eq ((p.prodEquivOfIsCompl q h).symm x).2 0) (Membership.mem p x)
  -/
  conv_rhs => rw [← (prodEquivOfIsCompl p q h).apply_symm_apply x]
  rw [coe_prodEquivOfIsCompl', Submodule.add_mem_iff_right _ (Submodule.coe_mem _),
    mem_left_iff_eq_zero_of_disjoint h.disjoint]


@[simp]
theorem prodComm_trans_prodEquivOfIsCompl (h : IsCompl p q) :
    LinearEquiv.prodComm R q p ≪≫ₗ prodEquivOfIsCompl p q h = prodEquivOfIsCompl q p h.symm :=
  LinearEquiv.ext fun _ => add_comm _ _


/-- Projection to a submodule along a complement.

See also `LinearMap.linearProjOfIsCompl`. -/
def linearProjOfIsCompl (h : IsCompl p q) : E →ₗ[R] p :=
  LinearMap.fst R p q ∘ₗ ↑(prodEquivOfIsCompl p q h).symm


@[simp]
theorem linearProjOfIsCompl_apply_left (h : IsCompl p q) (x : p) :
                                          /-
                                            R : Type u_1
                                            inst✝² : Ring R
                                            E : Type u_2
                                            inst✝¹ : AddCommGroup E
                                            inst✝ : Module R E
                                            p q : Submodule R E
                                            h : IsCompl p q
                                            x : Subtype fun x => Membership.mem p x
                                            ⊢ Eq ((p.linearProjOfIsCompl q h) ↑x) x
                                          -/
    linearProjOfIsCompl p q h x = x := by simp [linearProjOfIsCompl]
                                          /-
                                            🎉 no goals
                                          -/


@[simp]
theorem linearProjOfIsCompl_range (h : IsCompl p q) : range (linearProjOfIsCompl p q h) = ⊤ :=
  range_eq_of_proj (linearProjOfIsCompl_apply_left h)


@[simp]
theorem linearProjOfIsCompl_apply_eq_zero_iff (h : IsCompl p q) {x : E} :
                                                  /-
                                                    R : Type u_1
                                                    inst✝² : Ring R
                                                    E : Type u_2
                                                    inst✝¹ : AddCommGroup E
                                                    inst✝ : Module R E
                                                    p q : Submodule R E
                                                    h : IsCompl p q
                                                    x : E
                                                    ⊢ Iff (Eq ((p.linearProjOfIsCompl q h) x) 0) (Membership.mem q x)
                                                  -/
    linearProjOfIsCompl p q h x = 0 ↔ x ∈ q := by simp [linearProjOfIsCompl]
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem linearProjOfIsCompl_apply_right' (h : IsCompl p q) (x : E) (hx : x ∈ q) :
    linearProjOfIsCompl p q h x = 0 :=
  (linearProjOfIsCompl_apply_eq_zero_iff h).2 hx


@[simp]
theorem linearProjOfIsCompl_apply_right (h : IsCompl p q) (x : q) :
    linearProjOfIsCompl p q h x = 0 :=
  linearProjOfIsCompl_apply_right' h x x.2


@[simp]
theorem linearProjOfIsCompl_ker (h : IsCompl p q) : ker (linearProjOfIsCompl p q h) = q :=
  ext fun _ => mem_ker.trans (linearProjOfIsCompl_apply_eq_zero_iff h)


theorem linearProjOfIsCompl_comp_subtype (h : IsCompl p q) :
    (linearProjOfIsCompl p q h).comp p.subtype = LinearMap.id :=
  LinearMap.ext <| linearProjOfIsCompl_apply_left h


theorem linearProjOfIsCompl_idempotent (h : IsCompl p q) (x : E) :
    linearProjOfIsCompl p q h (linearProjOfIsCompl p q h x) = linearProjOfIsCompl p q h x :=
  linearProjOfIsCompl_apply_left h _


theorem existsUnique_add_of_isCompl_prod (hc : IsCompl p q) (x : E) :
    ∃! u : p × q, (u.fst : E) + u.snd = x :=
  (prodEquivOfIsCompl _ _ hc).toEquiv.bijective.existsUnique _


theorem existsUnique_add_of_isCompl (hc : IsCompl p q) (x : E) :
    ∃ (u : p) (v : q), (u : E) + v = x ∧ ∀ (r : p) (s : q), (r : E) + s = x → r = u ∧ s = v :=
  let ⟨u, hu₁, hu₂⟩ := existsUnique_add_of_isCompl_prod hc x
  ⟨u.1, u.2, hu₁, fun r s hrs => Prod.eq_iff_fst_eq_snd_eq.1 (hu₂ ⟨r, s⟩ hrs)⟩


theorem linear_proj_add_linearProjOfIsCompl_eq_self (hpq : IsCompl p q) (x : E) :
    (p.linearProjOfIsCompl q hpq x + q.linearProjOfIsCompl p hpq.symm x : E) = x := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p q : Submodule R E
    hpq : IsCompl p q
    x : E
    ⊢ Eq (HAdd.hAdd ↑((p.linearProjOfIsCompl q hpq) x) ↑((q.linearProjOfIsCompl p  …
  -/
  dsimp only [linearProjOfIsCompl]
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p q : Submodule R E
    hpq : IsCompl p q
    x : E
    ⊢ Eq (HAdd.hAdd ↑(((LinearMap.fst R (Subtype fun x => Membership.mem p x) (Sub …
  -/
  rw [← prodComm_trans_prodEquivOfIsCompl _ _ hpq]
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p q : Submodule R E
    hpq : IsCompl p q
    x : E
    ⊢ Eq (HAdd.hAdd ↑(((LinearMap.fst R (Subtype fun x => Membership.mem p x) (Sub …
  -/
  exact (prodEquivOfIsCompl _ _ hpq).apply_symm_apply x
  /-
    🎉 no goals
  -/


/-- Projection to the image of an injection along a complement.

This has an advantage over `Submodule.linearProjOfIsCompl` in that it allows the user better
definitional control over the type. -/
def linearProjOfIsCompl {F : Type*} [AddCommGroup F] [Module R F]
    (i : F →ₗ[R] E) (hi : Function.Injective i)
    (h : IsCompl (LinearMap.range i) q) : E →ₗ[R] F :=
  (LinearEquiv.ofInjective i hi).symm ∘ₗ (LinearMap.range i).linearProjOfIsCompl q h


@[simp]
theorem linearProjOfIsCompl_apply_left {F : Type*} [AddCommGroup F] [Module R F]
    (i : F →ₗ[R] E) (hi : Function.Injective i)
    (h : IsCompl (LinearMap.range i) q) (x : F) :
    linearProjOfIsCompl q i hi h (i x) = x := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    q : Submodule R E
    F : Type u_7
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    i : LinearMap (RingHom.id R) F E
    hi : Function.Injective ⇑i
    h : IsCompl (LinearMap.range i) q
    x : F
    ⊢ Eq ((LinearMap.linearProjOfIsCompl q i hi h) (i x)) x
  -/
  let ix : LinearMap.range i := ⟨i x, mem_range_self i x⟩
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    q : Submodule R E
    F : Type u_7
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    i : LinearMap (RingHom.id R) F E
    hi : Function.Injective ⇑i
    h : IsCompl (LinearMap.range i) q
    x : F
    ix : Subtype fun x => Membership.mem (LinearMap.range i) x := ⟨i x, ⋯⟩
    ⊢ Eq ((LinearMap.linearProjOfIsCompl q i hi h) (i x)) x
  -/
  change linearProjOfIsCompl q i hi h ix = x
  rw [linearProjOfIsCompl, coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    LinearEquiv.symm_apply_eq, Submodule.linearProjOfIsCompl_apply_left, Subtype.ext_iff,
    LinearEquiv.ofInjective_apply]


/-- Given linear maps `φ` and `ψ` from complement submodules, `LinearMap.ofIsCompl` is
the induced linear map over the entire module. -/
def ofIsCompl {p q : Submodule R E} (h : IsCompl p q) (φ : p →ₗ[R] F) (ψ : q →ₗ[R] F) : E →ₗ[R] F :=
  LinearMap.coprod φ ψ ∘ₗ ↑(Submodule.prodEquivOfIsCompl _ _ h).symm


@[simp]
theorem ofIsCompl_left_apply (h : IsCompl p q) {φ : p →ₗ[R] F} {ψ : q →ₗ[R] F} (u : p) :
                                        /-
                                          R : Type u_1
                                          inst✝⁴ : Ring R
                                          E : Type u_2
                                          inst✝³ : AddCommGroup E
                                          inst✝² : Module R E
                                          F : Type u_3
                                          inst✝¹ : AddCommGroup F
                                          inst✝ : Module R F
                                          p q : Submodule R E
                                          h : IsCompl p q
                                          φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
                                          ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
                                          u : Subtype fun x => Membership.mem p x
                                          ⊢ Eq ((LinearMap.ofIsCompl h φ ψ) ↑u) (φ u)
                                        -/
    ofIsCompl h φ ψ (u : E) = φ u := by simp [ofIsCompl]
                                        /-
                                          🎉 no goals
                                        -/


@[simp]
theorem ofIsCompl_right_apply (h : IsCompl p q) {φ : p →ₗ[R] F} {ψ : q →ₗ[R] F} (v : q) :
                                        /-
                                          R : Type u_1
                                          inst✝⁴ : Ring R
                                          E : Type u_2
                                          inst✝³ : AddCommGroup E
                                          inst✝² : Module R E
                                          F : Type u_3
                                          inst✝¹ : AddCommGroup F
                                          inst✝ : Module R F
                                          p q : Submodule R E
                                          h : IsCompl p q
                                          φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
                                          ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
                                          v : Subtype fun x => Membership.mem q x
                                          ⊢ Eq ((LinearMap.ofIsCompl h φ ψ) ↑v) (ψ v)
                                        -/
    ofIsCompl h φ ψ (v : E) = ψ v := by simp [ofIsCompl]
                                        /-
                                          🎉 no goals
                                        -/


theorem ofIsCompl_eq (h : IsCompl p q) {φ : p →ₗ[R] F} {ψ : q →ₗ[R] F} {χ : E →ₗ[R] F}
    (hφ : ∀ u, φ u = χ u) (hψ : ∀ u, ψ u = χ u) : ofIsCompl h φ ψ = χ := by
  /-
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    p q : Submodule R E
    h : IsCompl p q
    φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
    χ : LinearMap (RingHom.id R) E F
    hφ : ∀ (u : Subtype fun x => Membership.mem p x), Eq (φ u) (χ ↑u)
    hψ : ∀ (u : Subtype fun x => Membership.mem q x), Eq (ψ u) (χ ↑u)
    ⊢ Eq (LinearMap.ofIsCompl h φ ψ) χ
  -/
  ext x
  /-
    case h
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    p q : Submodule R E
    h : IsCompl p q
    φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
    χ : LinearMap (RingHom.id R) E F
    hφ : ∀ (u : Subtype fun x => Membership.mem p x), Eq (φ u) (χ ↑u)
    hψ : ∀ (u : Subtype fun x => Membership.mem q x), Eq (ψ u) (χ ↑u)
    x : E
    ⊢ Eq ((LinearMap.ofIsCompl h φ ψ) x) (χ x)
  -/
  obtain ⟨_, _, rfl, _⟩ := existsUnique_add_of_isCompl h x
  /-
    case h.intro.intro.intro
    R : Type u_1
    inst✝⁴ : Ring R
    E : Type u_2
    inst✝³ : AddCommGroup E
    inst✝² : Module R E
    F : Type u_3
    inst✝¹ : AddCommGroup F
    inst✝ : Module R F
    p q : Submodule R E
    h : IsCompl p q
    φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
    ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
    χ : LinearMap (RingHom.id R) E F
    hφ : ∀ (u : Subtype fun x => Membership.mem p x), Eq (φ u) (χ ↑u)
    hψ : ∀ (u : Subtype fun x => Membership.mem q x), Eq (ψ u) (χ ↑u)
    w✝¹ : Subtype fun x => Membership.mem p x
    w✝ : Subtype fun x => Membership.mem q x
    right✝ : ∀ (r : Subtype fun x => Membership.mem p x) (s : Subtype fun x => Mem …
    ⊢ Eq ((LinearMap.ofIsCompl h φ ψ) (HAdd.hAdd ↑w✝¹ ↑w✝)) (χ (HAdd.hAdd ↑w✝¹ ↑w✝))
  -/
  simp [ofIsCompl, hφ, hψ]
  /-
    🎉 no goals
  -/


theorem ofIsCompl_eq' (h : IsCompl p q) {φ : p →ₗ[R] F} {ψ : q →ₗ[R] F} {χ : E →ₗ[R] F}
    (hφ : φ = χ.comp p.subtype) (hψ : ψ = χ.comp q.subtype) : ofIsCompl h φ ψ = χ :=
  ofIsCompl_eq h (fun _ => hφ.symm ▸ rfl) fun _ => hψ.symm ▸ rfl


@[simp]
theorem ofIsCompl_zero (h : IsCompl p q) : (ofIsCompl h 0 0 : E →ₗ[R] F) = 0 :=
  ofIsCompl_eq _ (fun _ => rfl) fun _ => rfl


@[simp]
theorem ofIsCompl_add (h : IsCompl p q) {φ₁ φ₂ : p →ₗ[R] F} {ψ₁ ψ₂ : q →ₗ[R] F} :
    ofIsCompl h (φ₁ + φ₂) (ψ₁ + ψ₂) = ofIsCompl h φ₁ ψ₁ + ofIsCompl h φ₂ ψ₂ :=
                     /-
                       R : Type u_1
                       inst✝⁴ : Ring R
                       E : Type u_2
                       inst✝³ : AddCommGroup E
                       inst✝² : Module R E
                       F : Type u_3
                       inst✝¹ : AddCommGroup F
                       inst✝ : Module R F
                       p q : Submodule R E
                       h : IsCompl p q
                       φ₁ φ₂ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
                       ψ₁ ψ₂ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
                       ⊢ ∀ (u : Subtype fun x => Membership.mem p x), Eq ((HAdd.hAdd φ₁ φ₂) u) ((HAdd …
                     -/
                     /-
                       🎉 no goals
                     -/
  ofIsCompl_eq _ (by simp) (by simp)
                               /-
                                 🎉 no goals
                               -/


@[simp]
theorem ofIsCompl_smul {R : Type*} [CommRing R] {E : Type*} [AddCommGroup E] [Module R E]
    {F : Type*} [AddCommGroup F] [Module R F] {p q : Submodule R E} (h : IsCompl p q)
    {φ : p →ₗ[R] F} {ψ : q →ₗ[R] F} (c : R) : ofIsCompl h (c • φ) (c • ψ) = c • ofIsCompl h φ ψ :=
                     /-
                       R : Type u_7
                       inst✝⁴ : CommRing R
                       E : Type u_8
                       inst✝³ : AddCommGroup E
                       inst✝² : Module R E
                       F : Type u_9
                       inst✝¹ : AddCommGroup F
                       inst✝ : Module R F
                       p q : Submodule R E
                       h : IsCompl p q
                       φ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem p x) F
                       ψ : LinearMap (RingHom.id R) (Subtype fun x => Membership.mem q x) F
                       c : R
                       ⊢ ∀ (u : Subtype fun x => Membership.mem p x), Eq ((HSMul.hSMul c φ) u) ((HSMu …
                     -/
                     /-
                       🎉 no goals
                     -/
  ofIsCompl_eq _ (by simp) (by simp)
                               /-
                                 🎉 no goals
                               -/


/-- The linear map from `(p →ₗ[R₁] F) × (q →ₗ[R₁] F)` to `E →ₗ[R₁] F`. -/
def ofIsComplProd {p q : Submodule R₁ E} (h : IsCompl p q) :
    (p →ₗ[R₁] F) × (q →ₗ[R₁] F) →ₗ[R₁] E →ₗ[R₁] F where
  toFun φ := ofIsCompl h φ.1 φ.2
                 /-
                   R : Type u_1
                   inst✝¹² : Ring R
                   E : Type u_2
                   inst✝¹¹ : AddCommGroup E
                   inst✝¹⁰ : Module R E
                   F : Type u_3
                   inst✝⁹ : AddCommGroup F
                   inst✝⁸ : Module R F
                   G : Type u_4
                   inst✝⁷ : AddCommGroup G
                   inst✝⁶ : Module R G
                   p✝ q✝ : Submodule R E
                   S : Type u_5
                   inst✝⁵ : Semiring S
                   M : Type u_6
                   inst✝⁴ : AddCommMonoid M
                   inst✝³ : Module S M
                   m : Submodule S M
                   R₁ : Type u_7
                   inst✝² : CommRing R₁
                   inst✝¹ : Module R₁ E
                   inst✝ : Module R₁ F
                   p q : Submodule R₁ E
                   h : IsCompl p q
                   ⊢ ∀ (x y : Prod (LinearMap (RingHom.id R₁) (Subtype fun x => Membership.mem p  …
                 -/
  map_add' := by intro φ ψ; dsimp only; rw [Prod.snd_add, Prod.fst_add, ofIsCompl_add]
                                        /-
                                          🎉 no goals
                                        -/
                  /-
                    R : Type u_1
                    inst✝¹² : Ring R
                    E : Type u_2
                    inst✝¹¹ : AddCommGroup E
                    inst✝¹⁰ : Module R E
                    F : Type u_3
                    inst✝⁹ : AddCommGroup F
                    inst✝⁸ : Module R F
                    G : Type u_4
                    inst✝⁷ : AddCommGroup G
                    inst✝⁶ : Module R G
                    p✝ q✝ : Submodule R E
                    S : Type u_5
                    inst✝⁵ : Semiring S
                    M : Type u_6
                    inst✝⁴ : AddCommMonoid M
                    inst✝³ : Module S M
                    m : Submodule S M
                    R₁ : Type u_7
                    inst✝² : CommRing R₁
                    inst✝¹ : Module R₁ E
                    inst✝ : Module R₁ F
                    p q : Submodule R₁ E
                    h : IsCompl p q
                    ⊢ ∀ (m : R₁) (x : Prod (LinearMap (RingHom.id R₁) (Subtype fun x => Membership …
                  -/
  map_smul' := by intro c φ; simp [Prod.smul_snd, Prod.smul_fst, ofIsCompl_smul]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem ofIsComplProd_apply {p q : Submodule R₁ E} (h : IsCompl p q)
    (φ : (p →ₗ[R₁] F) × (q →ₗ[R₁] F)) : ofIsComplProd h φ = ofIsCompl h φ.1 φ.2 :=
  rfl


/-- The natural linear equivalence between `(p →ₗ[R₁] F) × (q →ₗ[R₁] F)` and `E →ₗ[R₁] F`. -/
def ofIsComplProdEquiv {p q : Submodule R₁ E} (h : IsCompl p q) :
    ((p →ₗ[R₁] F) × (q →ₗ[R₁] F)) ≃ₗ[R₁] E →ₗ[R₁] F :=
  { ofIsComplProd h with
    invFun := fun φ => ⟨φ.domRestrict p, φ.domRestrict q⟩
    left_inv := fun φ ↦ by
      /-
        R : Type u_1
        inst✝¹² : Ring R
        E : Type u_2
        inst✝¹¹ : AddCommGroup E
        inst✝¹⁰ : Module R E
        F : Type u_3
        inst✝⁹ : AddCommGroup F
        inst✝⁸ : Module R F
        G : Type u_4
        inst✝⁷ : AddCommGroup G
        inst✝⁶ : Module R G
        p✝ q✝ : Submodule R E
        S : Type u_5
        inst✝⁵ : Semiring S
        M : Type u_6
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module S M
        m : Submodule S M
        R₁ : Type u_7
        inst✝² : CommRing R₁
        inst✝¹ : Module R₁ E
        inst✝ : Module R₁ F
        p q : Submodule R₁ E
        h : IsCompl p q
        φ : Prod (LinearMap (RingHom.id R₁) (Subtype fun x => Membership.mem p x) F) ( …
        ⊢ Eq ((fun φ => { fst := φ.domRestrict p, snd := φ.domRestrict q }) (__src✝.to …
      -/
      ext x
        /-
          case fst.h
          R : Type u_1
          inst✝¹² : Ring R
          E : Type u_2
          inst✝¹¹ : AddCommGroup E
          inst✝¹⁰ : Module R E
          F : Type u_3
          inst✝⁹ : AddCommGroup F
          inst✝⁸ : Module R F
          G : Type u_4
          inst✝⁷ : AddCommGroup G
          inst✝⁶ : Module R G
          p✝ q✝ : Submodule R E
          S : Type u_5
          inst✝⁵ : Semiring S
          M : Type u_6
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module S M
          m : Submodule S M
          R₁ : Type u_7
          inst✝² : CommRing R₁
          inst✝¹ : Module R₁ E
          inst✝ : Module R₁ F
          p q : Submodule R₁ E
          h : IsCompl p q
          φ : Prod (LinearMap (RingHom.id R₁) (Subtype fun x => Membership.mem p x) F) ( …
          x : Subtype fun x => Membership.mem p x
          ⊢ Eq (((fun φ => { fst := φ.domRestrict p, snd := φ.domRestrict q }) (__src✝.t …
        -/
      · exact ofIsCompl_left_apply h x
        /-
          🎉 no goals
        -/
        /-
          case snd.h
          R : Type u_1
          inst✝¹² : Ring R
          E : Type u_2
          inst✝¹¹ : AddCommGroup E
          inst✝¹⁰ : Module R E
          F : Type u_3
          inst✝⁹ : AddCommGroup F
          inst✝⁸ : Module R F
          G : Type u_4
          inst✝⁷ : AddCommGroup G
          inst✝⁶ : Module R G
          p✝ q✝ : Submodule R E
          S : Type u_5
          inst✝⁵ : Semiring S
          M : Type u_6
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module S M
          m : Submodule S M
          R₁ : Type u_7
          inst✝² : CommRing R₁
          inst✝¹ : Module R₁ E
          inst✝ : Module R₁ F
          p q : Submodule R₁ E
          h : IsCompl p q
          φ : Prod (LinearMap (RingHom.id R₁) (Subtype fun x => Membership.mem p x) F) ( …
          x : Subtype fun x => Membership.mem q x
          ⊢ Eq (((fun φ => { fst := φ.domRestrict p, snd := φ.domRestrict q }) (__src✝.t …
        -/
      · exact ofIsCompl_right_apply h x
        /-
          🎉 no goals
        -/
    right_inv := fun φ ↦ by
      /-
        R : Type u_1
        inst✝¹² : Ring R
        E : Type u_2
        inst✝¹¹ : AddCommGroup E
        inst✝¹⁰ : Module R E
        F : Type u_3
        inst✝⁹ : AddCommGroup F
        inst✝⁸ : Module R F
        G : Type u_4
        inst✝⁷ : AddCommGroup G
        inst✝⁶ : Module R G
        p✝ q✝ : Submodule R E
        S : Type u_5
        inst✝⁵ : Semiring S
        M : Type u_6
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module S M
        m : Submodule S M
        R₁ : Type u_7
        inst✝² : CommRing R₁
        inst✝¹ : Module R₁ E
        inst✝ : Module R₁ F
        p q : Submodule R₁ E
        h : IsCompl p q
        φ : LinearMap (RingHom.id R₁) E F
        ⊢ Eq (__src✝.toFun ((fun φ => { fst := φ.domRestrict p, snd := φ.domRestrict q …
      -/
      ext x
      /-
        case h
        R : Type u_1
        inst✝¹² : Ring R
        E : Type u_2
        inst✝¹¹ : AddCommGroup E
        inst✝¹⁰ : Module R E
        F : Type u_3
        inst✝⁹ : AddCommGroup F
        inst✝⁸ : Module R F
        G : Type u_4
        inst✝⁷ : AddCommGroup G
        inst✝⁶ : Module R G
        p✝ q✝ : Submodule R E
        S : Type u_5
        inst✝⁵ : Semiring S
        M : Type u_6
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module S M
        m : Submodule S M
        R₁ : Type u_7
        inst✝² : CommRing R₁
        inst✝¹ : Module R₁ E
        inst✝ : Module R₁ F
        p q : Submodule R₁ E
        h : IsCompl p q
        φ : LinearMap (RingHom.id R₁) E F
        x : E
        ⊢ Eq ((__src✝.toFun ((fun φ => { fst := φ.domRestrict p, snd := φ.domRestrict  …
      -/
      obtain ⟨a, b, hab, _⟩ := existsUnique_add_of_isCompl h x
      /-
        case h.intro.intro.intro
        R : Type u_1
        inst✝¹² : Ring R
        E : Type u_2
        inst✝¹¹ : AddCommGroup E
        inst✝¹⁰ : Module R E
        F : Type u_3
        inst✝⁹ : AddCommGroup F
        inst✝⁸ : Module R F
        G : Type u_4
        inst✝⁷ : AddCommGroup G
        inst✝⁶ : Module R G
        p✝ q✝ : Submodule R E
        S : Type u_5
        inst✝⁵ : Semiring S
        M : Type u_6
        inst✝⁴ : AddCommMonoid M
        inst✝³ : Module S M
        m : Submodule S M
        R₁ : Type u_7
        inst✝² : CommRing R₁
        inst✝¹ : Module R₁ E
        inst✝ : Module R₁ F
        p q : Submodule R₁ E
        h : IsCompl p q
        φ : LinearMap (RingHom.id R₁) E F
        x : E
        a : Subtype fun x => Membership.mem p x
        b : Subtype fun x => Membership.mem q x
        hab : Eq (HAdd.hAdd ↑a ↑b) x
        right✝ : ∀ (r : Subtype fun x => Membership.mem p x) (s : Subtype fun x => Mem …
        ⊢ Eq ((__src✝.toFun ((fun φ => { fst := φ.domRestrict p, snd := φ.domRestrict  …
      -/
      rw [← hab]; simp }
                  /-
                    🎉 no goals
                  -/


@[simp, nolint simpNF] -- Porting note: linter claims that LHS doesn't simplify, but it does
theorem linearProjOfIsCompl_of_proj (f : E →ₗ[R] p) (hf : ∀ x : p, f x = x) :
    p.linearProjOfIsCompl (ker f) (isCompl_of_proj hf) = f := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    ⊢ Eq (p.linearProjOfIsCompl (LinearMap.ker f) ⋯) f
  -/
  ext x
  /-
    case h.a
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    x : E
    ⊢ Eq ↑((p.linearProjOfIsCompl (LinearMap.ker f) ⋯) x) ↑(f x)
  -/
  have : x ∈ p ⊔ (ker f) := by simp only [(isCompl_of_proj hf).sup_eq_top, mem_top]
  /-
    case h.a
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    x : E
    this : Membership.mem (Max.max p (LinearMap.ker f)) x
    ⊢ Eq ↑((p.linearProjOfIsCompl (LinearMap.ker f) ⋯) x) ↑(f x)
  -/
  rcases mem_sup'.1 this with ⟨x, y, rfl⟩
  /-
    case h.a.intro.intro
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E (Subtype fun x => Membership.mem p x)
    hf : ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
    x : Subtype fun x => Membership.mem p x
    y : Subtype fun x => Membership.mem (LinearMap.ker f) x
    this : Membership.mem (Max.max p (LinearMap.ker f)) (HAdd.hAdd ↑x ↑y)
    ⊢ Eq ↑((p.linearProjOfIsCompl (LinearMap.ker f) ⋯) (HAdd.hAdd ↑x ↑y)) ↑(f (HAd …
  -/
  simp [hf]
  /-
    🎉 no goals
  -/


/-- If `f : E →ₗ[R] F` and `g : E →ₗ[R] G` are two surjective linear maps and
their kernels are complement of each other, then `x ↦ (f x, g x)` defines
a linear equivalence `E ≃ₗ[R] F × G`. -/
def equivProdOfSurjectiveOfIsCompl (f : E →ₗ[R] F) (g : E →ₗ[R] G) (hf : range f = ⊤)
    (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) : E ≃ₗ[R] F × G :=
  LinearEquiv.ofBijective (f.prod g)
        /-
          R : Type u_1
          inst✝⁹ : Ring R
          E : Type u_2
          inst✝⁸ : AddCommGroup E
          inst✝⁷ : Module R E
          F : Type u_3
          inst✝⁶ : AddCommGroup F
          inst✝⁵ : Module R F
          G : Type u_4
          inst✝⁴ : AddCommGroup G
          inst✝³ : Module R G
          p q : Submodule R E
          S : Type u_5
          inst✝² : Semiring S
          M : Type u_6
          inst✝¹ : AddCommMonoid M
          inst✝ : Module S M
          m : Submodule S M
          f : LinearMap (RingHom.id R) E F
          g : LinearMap (RingHom.id R) E G
          hf : Eq (LinearMap.range f) Top.top
          hg : Eq (LinearMap.range g) Top.top
          hfg : IsCompl (LinearMap.ker f) (LinearMap.ker g)
          ⊢ Function.Injective ⇑(f.prod g)
        -/
    ⟨by simp [← ker_eq_bot, hfg.inf_eq_bot], by
        /-
          🎉 no goals
        -/
      /-
        R : Type u_1
        inst✝⁹ : Ring R
        E : Type u_2
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module R E
        F : Type u_3
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module R F
        G : Type u_4
        inst✝⁴ : AddCommGroup G
        inst✝³ : Module R G
        p q : Submodule R E
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        m : Submodule S M
        f : LinearMap (RingHom.id R) E F
        g : LinearMap (RingHom.id R) E G
        hf : Eq (LinearMap.range f) Top.top
        hg : Eq (LinearMap.range g) Top.top
        hfg : IsCompl (LinearMap.ker f) (LinearMap.ker g)
        ⊢ Function.Surjective ⇑(f.prod g)
      -/
      rw [← range_eq_top]
      /-
        R : Type u_1
        inst✝⁹ : Ring R
        E : Type u_2
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module R E
        F : Type u_3
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module R F
        G : Type u_4
        inst✝⁴ : AddCommGroup G
        inst✝³ : Module R G
        p q : Submodule R E
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        m : Submodule S M
        f : LinearMap (RingHom.id R) E F
        g : LinearMap (RingHom.id R) E G
        hf : Eq (LinearMap.range f) Top.top
        hg : Eq (LinearMap.range g) Top.top
        hfg : IsCompl (LinearMap.ker f) (LinearMap.ker g)
        ⊢ Eq (LinearMap.range (f.prod g)) Top.top
      -/
      simp [range_prod_eq hfg.sup_eq_top, *]⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem coe_equivProdOfSurjectiveOfIsCompl {f : E →ₗ[R] F} {g : E →ₗ[R] G} (hf : range f = ⊤)
    (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) :
    (equivProdOfSurjectiveOfIsCompl f g hf hg hfg : E →ₗ[R] F × G) = f.prod g := rfl


@[simp]
theorem equivProdOfSurjectiveOfIsCompl_apply {f : E →ₗ[R] F} {g : E →ₗ[R] G} (hf : range f = ⊤)
    (hg : range g = ⊤) (hfg : IsCompl (ker f) (ker g)) (x : E) :
    equivProdOfSurjectiveOfIsCompl f g hf hg hfg x = (f x, g x) := rfl


/-- Equivalence between submodules `q` such that `IsCompl p q` and linear maps `f : E →ₗ[R] p`
such that `∀ x : p, f x = x`. -/
def isComplEquivProj : { q // IsCompl p q } ≃ { f : E →ₗ[R] p // ∀ x : p, f x = x } where
  toFun q := ⟨linearProjOfIsCompl p q q.2, linearProjOfIsCompl_apply_left q.2⟩
  invFun f := ⟨ker (f : E →ₗ[R] p), isCompl_of_proj f.2⟩
                                /-
                                  R : Type u_1
                                  inst✝⁹ : Ring R
                                  E : Type u_2
                                  inst✝⁸ : AddCommGroup E
                                  inst✝⁷ : Module R E
                                  F : Type u_3
                                  inst✝⁶ : AddCommGroup F
                                  inst✝⁵ : Module R F
                                  G : Type u_4
                                  inst✝⁴ : AddCommGroup G
                                  inst✝³ : Module R G
                                  p q✝ : Submodule R E
                                  S : Type u_5
                                  inst✝² : Semiring S
                                  M : Type u_6
                                  inst✝¹ : AddCommMonoid M
                                  inst✝ : Module S M
                                  m : Submodule S M
                                  x✝ : Subtype fun q => IsCompl p q
                                  q : Submodule R E
                                  hq : IsCompl p q
                                  ⊢ Eq ((fun f => ⟨LinearMap.ker ↑f, ⋯⟩) ((fun q => ⟨p.linearProjOfIsCompl ↑q ⋯, …
                                -/
  left_inv := fun ⟨q, hq⟩ => by simp only [linearProjOfIsCompl_ker, Subtype.coe_mk]
                                /-
                                  🎉 no goals
                                -/
  right_inv := fun ⟨f, hf⟩ => Subtype.eq <| f.linearProjOfIsCompl_of_proj hf


@[simp]
theorem coe_isComplEquivProj_apply (q : { q // IsCompl p q }) :
    (p.isComplEquivProj q : E →ₗ[R] p) = linearProjOfIsCompl p q q.2 := rfl


@[simp]
theorem coe_isComplEquivProj_symm_apply (f : { f : E →ₗ[R] p // ∀ x : p, f x = x }) :
    (p.isComplEquivProj.symm f : Submodule R E) = ker (f : E →ₗ[R] p) := rfl


/-- The idempotent endomorphisms of a module with range equal to a submodule are in 1-1
correspondence with linear maps to the submodule that restrict to the identity on the submodule. -/
@[simps] def isIdempotentElemEquiv :
    { f : Module.End R E // IsIdempotentElem f ∧ range f = p } ≃
    { f : E →ₗ[R] p // ∀ x : p, f x = x } where
                                           /-
                                             R : Type u_1
                                             inst✝⁹ : Ring R
                                             E : Type u_2
                                             inst✝⁸ : AddCommGroup E
                                             inst✝⁷ : Module R E
                                             F : Type u_3
                                             inst✝⁶ : AddCommGroup F
                                             inst✝⁵ : Module R F
                                             G : Type u_4
                                             inst✝⁴ : AddCommGroup G
                                             inst✝³ : Module R G
                                             p q : Submodule R E
                                             S : Type u_5
                                             inst✝² : Semiring S
                                             M : Type u_6
                                             inst✝¹ : AddCommMonoid M
                                             inst✝ : Module S M
                                             m : Submodule S M
                                             f : Subtype fun f => And (IsIdempotentElem f) (Eq (LinearMap.range f) p)
                                             x : E
                                             ⊢ Membership.mem p (↑f x)
                                           -/
  toFun f := ⟨f.1.codRestrict _ fun x ↦ by simp_rw [← f.2.2]; exact mem_range_self f.1 x,
                                                              /-
                                                                🎉 no goals
                                                              -/
    fun ⟨x, hx⟩ ↦ Subtype.ext <| by
      /-
        R : Type u_1
        inst✝⁹ : Ring R
        E : Type u_2
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module R E
        F : Type u_3
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module R F
        G : Type u_4
        inst✝⁴ : AddCommGroup G
        inst✝³ : Module R G
        p q : Submodule R E
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        m : Submodule S M
        f : Subtype fun f => And (IsIdempotentElem f) (Eq (LinearMap.range f) p)
        x✝ : Subtype fun x => Membership.mem p x
        x : E
        hx : Membership.mem p x
        ⊢ Eq ↑((LinearMap.codRestrict p ↑f ⋯) ↑⟨x, hx⟩) ↑⟨x, hx⟩
      -/
      obtain ⟨x, rfl⟩ := f.2.2.symm ▸ hx
      /-
        case intro
        R : Type u_1
        inst✝⁹ : Ring R
        E : Type u_2
        inst✝⁸ : AddCommGroup E
        inst✝⁷ : Module R E
        F : Type u_3
        inst✝⁶ : AddCommGroup F
        inst✝⁵ : Module R F
        G : Type u_4
        inst✝⁴ : AddCommGroup G
        inst✝³ : Module R G
        p q : Submodule R E
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        m : Submodule S M
        f : Subtype fun f => And (IsIdempotentElem f) (Eq (LinearMap.range f) p)
        x✝ : Subtype fun x => Membership.mem p x
        x : E
        hx : Membership.mem p (↑f x)
        ⊢ Eq ↑((LinearMap.codRestrict p ↑f ⋯) ↑⟨↑f x, hx⟩) ↑⟨↑f x, hx⟩
      -/
      exact DFunLike.congr_fun f.2.1 x⟩
      /-
        🎉 no goals
      -/
                                                          /-
                                                            R : Type u_1
                                                            inst✝⁹ : Ring R
                                                            E : Type u_2
                                                            inst✝⁸ : AddCommGroup E
                                                            inst✝⁷ : Module R E
                                                            F : Type u_3
                                                            inst✝⁶ : AddCommGroup F
                                                            inst✝⁵ : Module R F
                                                            G : Type u_4
                                                            inst✝⁴ : AddCommGroup G
                                                            inst✝³ : Module R G
                                                            p q : Submodule R E
                                                            S : Type u_5
                                                            inst✝² : Semiring S
                                                            M : Type u_6
                                                            inst✝¹ : AddCommMonoid M
                                                            inst✝ : Module S M
                                                            m : Submodule S M
                                                            f : Subtype fun f => ∀ (x : Subtype fun x => Membership.mem p x), Eq (f ↑x) x
                                                            x : E
                                                            ⊢ Eq ((HMul.hMul (p.subtype.comp ↑f) (p.subtype.comp ↑f)) x) ((p.subtype.comp  …
                                                          -/
  invFun f := ⟨p.subtype ∘ₗ f.1, LinearMap.ext fun x ↦ by simp [f.2], le_antisymm
                                                          /-
                                                            🎉 no goals
                                                          -/
    ((range_comp_le_range _ _).trans_eq p.range_subtype)
    fun x hx ↦ ⟨x, Subtype.ext_iff.1 <| f.2 ⟨x, hx⟩⟩⟩
  left_inv _ := rfl
  right_inv _ := rfl


/--
A linear endomorphism of a module `E` is a projection onto a submodule `p` if it sends every element
of `E` to `p` and fixes every element of `p`.
The definition allow more generally any `FunLike` type and not just linear maps, so that it can be
used for example with `ContinuousLinearMap` or `Matrix`.
-/
structure IsProj {F : Type*} [FunLike F M M] (f : F) : Prop where
  map_mem : ∀ x, f x ∈ m
  map_id : ∀ x ∈ m, f x = x


theorem isProj_iff_idempotent (f : M →ₗ[S] M) : (∃ p : Submodule S M, IsProj p f) ↔ f ∘ₗ f = f := by
  /-
    S : Type u_5
    inst✝² : Semiring S
    M : Type u_6
    inst✝¹ : AddCommMonoid M
    inst✝ : Module S M
    f : LinearMap (RingHom.id S) M M
    ⊢ Iff (Exists fun p => LinearMap.IsProj p f) (Eq (f.comp f) f)
  -/
  constructor
    /-
      case mp
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      ⊢ (Exists fun p => LinearMap.IsProj p f) → Eq (f.comp f) f
    -/
  · intro h
    /-
      case mp
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      h : Exists fun p => LinearMap.IsProj p f
      ⊢ Eq (f.comp f) f
    -/
    obtain ⟨p, hp⟩ := h
    /-
      case mp.intro
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      p : Submodule S M
      hp : LinearMap.IsProj p f
      ⊢ Eq (f.comp f) f
    -/
    ext x
    /-
      case mp.intro.h
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      p : Submodule S M
      hp : LinearMap.IsProj p f
      x : M
      ⊢ Eq ((f.comp f) x) (f x)
    -/
    rw [comp_apply]
    /-
      case mp.intro.h
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      p : Submodule S M
      hp : LinearMap.IsProj p f
      x : M
      ⊢ Eq (f (f x)) (f x)
    -/
    exact hp.map_id (f x) (hp.map_mem x)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      ⊢ Eq (f.comp f) f → Exists fun p => LinearMap.IsProj p f
    -/
  · intro h
    /-
      case mpr
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      h : Eq (f.comp f) f
      ⊢ Exists fun p => LinearMap.IsProj p f
    -/
    use range f
    /-
      case h
      S : Type u_5
      inst✝² : Semiring S
      M : Type u_6
      inst✝¹ : AddCommMonoid M
      inst✝ : Module S M
      f : LinearMap (RingHom.id S) M M
      h : Eq (f.comp f) f
      ⊢ LinearMap.IsProj (LinearMap.range f) f
    -/
    constructor
      /-
        case h.map_mem
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        f : LinearMap (RingHom.id S) M M
        h : Eq (f.comp f) f
        ⊢ ∀ (x : M), Membership.mem (LinearMap.range f) (f x)
      -/
    · intro x
      /-
        case h.map_mem
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        f : LinearMap (RingHom.id S) M M
        h : Eq (f.comp f) f
        x : M
        ⊢ Membership.mem (LinearMap.range f) (f x)
      -/
      exact mem_range_self f x
      /-
        🎉 no goals
      -/
      /-
        case h.map_id
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        f : LinearMap (RingHom.id S) M M
        h : Eq (f.comp f) f
        ⊢ ∀ (x : M), Membership.mem (LinearMap.range f) x → Eq (f x) x
      -/
    · intro x hx
      /-
        case h.map_id
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        f : LinearMap (RingHom.id S) M M
        h : Eq (f.comp f) f
        x : M
        hx : Membership.mem (LinearMap.range f) x
        ⊢ Eq (f x) x
      -/
      obtain ⟨y, hy⟩ := mem_range.1 hx
      /-
        case h.map_id.intro
        S : Type u_5
        inst✝² : Semiring S
        M : Type u_6
        inst✝¹ : AddCommMonoid M
        inst✝ : Module S M
        f : LinearMap (RingHom.id S) M M
        h : Eq (f.comp f) f
        x : M
        hx : Membership.mem (LinearMap.range f) x
        y : M
        hy : Eq (f y) x
        ⊢ Eq (f x) x
      -/
      rw [← hy, ← comp_apply, h]
      /-
        🎉 no goals
      -/


/-- Restriction of the codomain of a projection of onto a subspace `p` to `p` instead of the whole
space.
-/
def codRestrict {f : M →ₗ[S] M} (h : IsProj m f) : M →ₗ[S] m :=
  f.codRestrict m h.map_mem


@[simp]
theorem codRestrict_apply {f : M →ₗ[S] M} (h : IsProj m f) (x : M) : ↑(h.codRestrict x) = f x :=
  f.codRestrict_apply m x


@[simp]
theorem codRestrict_apply_cod {f : M →ₗ[S] M} (h : IsProj m f) (x : m) : h.codRestrict x = x := by
  /-
    S : Type u_5
    inst✝² : Semiring S
    M : Type u_6
    inst✝¹ : AddCommMonoid M
    inst✝ : Module S M
    m : Submodule S M
    f : LinearMap (RingHom.id S) M M
    h : LinearMap.IsProj m f
    x : Subtype fun x => Membership.mem m x
    ⊢ Eq (h.codRestrict ↑x) x
  -/
  ext
  /-
    case a
    S : Type u_5
    inst✝² : Semiring S
    M : Type u_6
    inst✝¹ : AddCommMonoid M
    inst✝ : Module S M
    m : Submodule S M
    f : LinearMap (RingHom.id S) M M
    h : LinearMap.IsProj m f
    x : Subtype fun x => Membership.mem m x
    ⊢ Eq ↑(h.codRestrict ↑x) ↑x
  -/
  rw [codRestrict_apply]
  /-
    case a
    S : Type u_5
    inst✝² : Semiring S
    M : Type u_6
    inst✝¹ : AddCommMonoid M
    inst✝ : Module S M
    m : Submodule S M
    f : LinearMap (RingHom.id S) M M
    h : LinearMap.IsProj m f
    x : Subtype fun x => Membership.mem m x
    ⊢ Eq (f ↑x) ↑x
  -/
  exact h.map_id x x.2
  /-
    🎉 no goals
  -/


theorem codRestrict_ker {f : M →ₗ[S] M} (h : IsProj m f) : ker h.codRestrict = ker f :=
  f.ker_codRestrict m _


theorem isCompl {f : E →ₗ[R] E} (h : IsProj p f) : IsCompl p (ker f) := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E E
    h : LinearMap.IsProj p f
    ⊢ IsCompl p (LinearMap.ker f)
  -/
  rw [← codRestrict_ker]
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E E
    h : LinearMap.IsProj p f
    ⊢ IsCompl p (LinearMap.ker (LinearMap.IsProj.codRestrict ?m.333839))
  -/
  exact isCompl_of_proj h.codRestrict_apply_cod
  /-
    🎉 no goals
  -/


theorem eq_conj_prod_map' {f : E →ₗ[R] E} (h : IsProj p f) :
    f = (p.prodEquivOfIsCompl (ker f) h.isCompl).toLinearMap ∘ₗ
        prodMap id 0 ∘ₗ (p.prodEquivOfIsCompl (ker f) h.isCompl).symm.toLinearMap := by
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E E
    h : LinearMap.IsProj p f
    ⊢ Eq f ((↑(p.prodEquivOfIsCompl (LinearMap.ker f) ⋯)).comp ((LinearMap.id.prod …
  -/
  rw [← LinearMap.comp_assoc, LinearEquiv.eq_comp_toLinearMap_symm]
  /-
    R : Type u_1
    inst✝² : Ring R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E E
    h : LinearMap.IsProj p f
    ⊢ Eq (f.comp ↑(p.prodEquivOfIsCompl (LinearMap.ker f) ⋯)) ((↑(p.prodEquivOfIsC …
  -/
  ext x
  · simp only [coe_prodEquivOfIsCompl, comp_apply, coe_inl, coprod_apply, coe_subtype,
      _root_.map_zero, add_zero, h.map_id x x.2, prodMap_apply, id_apply]
  · simp only [coe_prodEquivOfIsCompl, comp_apply, coe_inr, coprod_apply, _root_.map_zero,
      coe_subtype, zero_add, map_coe_ker, prodMap_apply, zero_apply, add_zero]


theorem IsProj.eq_conj_prodMap {f : E →ₗ[R] E} (h : IsProj p f) :
    f = (p.prodEquivOfIsCompl (ker f) h.isCompl).conj (prodMap id 0) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E E
    h : LinearMap.IsProj p f
    ⊢ Eq f ((p.prodEquivOfIsCompl (LinearMap.ker f) ⋯).conj (LinearMap.id.prodMap  …
  -/
  rw [LinearEquiv.conj_apply]
  /-
    R : Type u_1
    inst✝² : CommRing R
    E : Type u_2
    inst✝¹ : AddCommGroup E
    inst✝ : Module R E
    p : Submodule R E
    f : LinearMap (RingHom.id R) E E
    h : LinearMap.IsProj p f
    ⊢ Eq f (((↑(p.prodEquivOfIsCompl (LinearMap.ker f) ⋯)).comp (LinearMap.id.prod …
  -/
  exact h.eq_conj_prod_map'
  /-
    🎉 no goals
  -/


