/-- The natural `R`-linear map from a submodule of an `R`-module `M` to `M`. -/
protected def subtype : S' →ₗ[R] M where
  toFun := Subtype.val
  map_add' _ _ := rfl
  map_smul' _ _ := rfl


@[simp]
protected theorem coeSubtype : (SMulMemClass.subtype S' : S' → M) = Subtype.val :=
  rfl


/-- Embedding of a submodule `p` to the ambient space `M`. -/
protected def subtype : p →ₗ[R] M where
  toFun := Subtype.val
                 /-
                   G : Type u''
                   S : Type u'
                   R : Type u
                   M : Type v
                   ι : Type w
                   inst✝¹ : Semiring R
                   inst✝ : AddCommMonoid M
                   module_M : Module R M
                   p q : Submodule R M
                   r : R
                   x y : M
                   ⊢ ∀ (x y : Subtype fun x => Membership.mem p x), Eq (↑(HAdd.hAdd x y)) (HAdd.h …
                 -/
  map_add' := by simp [coe_smul]
                 /-
                   🎉 no goals
                 -/
                  /-
                    G : Type u''
                    S : Type u'
                    R : Type u
                    M : Type v
                    ι : Type w
                    inst✝¹ : Semiring R
                    inst✝ : AddCommMonoid M
                    module_M : Module R M
                    p q : Submodule R M
                    r : R
                    x y : M
                    ⊢ ∀ (m : R) (x : Subtype fun x => Membership.mem p x), Eq ({ toFun := Subtype. …
                  -/
  map_smul' := by simp [coe_smul]
                  /-
                    🎉 no goals
                  -/


theorem subtype_apply (x : p) : p.subtype x = x :=
  rfl


@[simp]
theorem coe_subtype : (Submodule.subtype p : p → M) = Subtype.val :=
  rfl


@[deprecated (since := "2024-09-27")] alias coeSubtype := coe_subtype


theorem injective_subtype : Injective p.subtype :=
  Subtype.coe_injective


/-- Note the `AddSubmonoid` version of this lemma is called `AddSubmonoid.coe_finset_sum`. -/
-- Porting note: removing the `@[simp]` attribute since it's literally `AddSubmonoid.coe_finset_sum`
theorem coe_sum (x : ι → p) (s : Finset ι) : ↑(∑ i ∈ s, x i) = ∑ i ∈ s, (x i : M) :=
  map_sum p.subtype _ _


/-- The action by a submodule is the action by the underlying module. -/
instance [AddAction M α] : AddAction p α :=
  AddAction.compHom _ p.subtype.toAddMonoidHom


/-- The restriction of a linear map `f : M → M₂` to a submodule `p ⊆ M` gives a linear map
`p → M₂`. -/
def domRestrict (f : M →ₛₗ[σ₁₂] M₂) (p : Submodule R M) : p →ₛₗ[σ₁₂] M₂ :=
  f.comp p.subtype


@[simp]
theorem domRestrict_apply (f : M →ₛₗ[σ₁₂] M₂) (p : Submodule R M) (x : p) :
    f.domRestrict p x = f x :=
  rfl


/-- A linear map `f : M₂ → M` whose values lie in a submodule `p ⊆ M` can be restricted to a
linear map M₂ → p. -/
def codRestrict (p : Submodule R₂ M₂) (f : M →ₛₗ[σ₁₂] M₂) (h : ∀ c, f c ∈ p) : M →ₛₗ[σ₁₂] p where
  toFun c := ⟨f c, h c⟩
                     /-
                       R : Type u_1
                       R₁ : Type u_2
                       R₂ : Type u_3
                       R₃ : Type u_4
                       M : Type u_5
                       M₁ : Type u_6
                       M₂ : Type u_7
                       M₃ : Type u_8
                       ι : Type u_9
                       inst✝¹¹ : Semiring R
                       inst✝¹⁰ : Semiring R₂
                       inst✝⁹ : Semiring R₃
                       inst✝⁸ : AddCommMonoid M
                       inst✝⁷ : AddCommMonoid M₁
                       inst✝⁶ : AddCommMonoid M₂
                       inst✝⁵ : AddCommMonoid M₃
                       inst✝⁴ : Module R M
                       inst✝³ : Module R M₁
                       inst✝² : Module R₂ M₂
                       inst✝¹ : Module R₃ M₃
                       σ₁₂ : RingHom R R₂
                       σ₂₃ : RingHom R₂ R₃
                       σ₁₃ : RingHom R R₃
                       inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                       f✝ : LinearMap σ₁₂ M M₂
                       g : LinearMap σ₂₃ M₂ M₃
                       p : Submodule R₂ M₂
                       f : LinearMap σ₁₂ M M₂
                       h : ∀ (c : M), Membership.mem p (f c)
                       x✝¹ x✝ : M
                       ⊢ Eq ((fun c => ⟨f c, ⋯⟩) (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd ((fun c => ⟨f c, ⋯⟩)  …
                     -/
  map_add' _ _ := by simp
                     /-
                       🎉 no goals
                     -/
                      /-
                        R : Type u_1
                        R₁ : Type u_2
                        R₂ : Type u_3
                        R₃ : Type u_4
                        M : Type u_5
                        M₁ : Type u_6
                        M₂ : Type u_7
                        M₃ : Type u_8
                        ι : Type u_9
                        inst✝¹¹ : Semiring R
                        inst✝¹⁰ : Semiring R₂
                        inst✝⁹ : Semiring R₃
                        inst✝⁸ : AddCommMonoid M
                        inst✝⁷ : AddCommMonoid M₁
                        inst✝⁶ : AddCommMonoid M₂
                        inst✝⁵ : AddCommMonoid M₃
                        inst✝⁴ : Module R M
                        inst✝³ : Module R M₁
                        inst✝² : Module R₂ M₂
                        inst✝¹ : Module R₃ M₃
                        σ₁₂ : RingHom R R₂
                        σ₂₃ : RingHom R₂ R₃
                        σ₁₃ : RingHom R R₃
                        inst✝ : RingHomCompTriple σ₁₂ σ₂₃ σ₁₃
                        f✝ : LinearMap σ₁₂ M M₂
                        g : LinearMap σ₂₃ M₂ M₃
                        p : Submodule R₂ M₂
                        f : LinearMap σ₁₂ M M₂
                        h : ∀ (c : M), Membership.mem p (f c)
                        x✝¹ : R
                        x✝ : M
                        ⊢ Eq ({ toFun := fun c => ⟨f c, ⋯⟩, map_add' := ⋯ }.toFun (HSMul.hSMul x✝¹ x✝) …
                      -/
  map_smul' _ _ := by simp
                      /-
                        🎉 no goals
                      -/


@[simp]
theorem codRestrict_apply (p : Submodule R₂ M₂) (f : M →ₛₗ[σ₁₂] M₂) {h} (x : M) :
    (codRestrict p f h x : M₂) = f x :=
  rfl


@[simp]
theorem comp_codRestrict (p : Submodule R₃ M₃) (h : ∀ b, g b ∈ p) :
    ((codRestrict p g h).comp f : M →ₛₗ[σ₁₃] p) = codRestrict p (g.comp f) fun _ => h _ :=
  ext fun _ => rfl


@[simp]
theorem subtype_comp_codRestrict (p : Submodule R₂ M₂) (h : ∀ b, f b ∈ p) :
    p.subtype.comp (codRestrict p f h) = f :=
  ext fun _ => rfl


/-- Restrict domain and codomain of a linear map. -/
def restrict (f : M →ₗ[R] M₁) {p : Submodule R M} {q : Submodule R M₁} (hf : ∀ x ∈ p, f x ∈ q) :
    p →ₗ[R] q :=
  (f.domRestrict p).codRestrict q <| SetLike.forall.2 hf


@[simp]
theorem restrict_coe_apply (f : M →ₗ[R] M₁) {p : Submodule R M} {q : Submodule R M₁}
    (hf : ∀ x ∈ p, f x ∈ q) (x : p) : ↑(f.restrict hf x) = f x :=
  rfl


theorem restrict_apply {f : M →ₗ[R] M₁} {p : Submodule R M} {q : Submodule R M₁}
    (hf : ∀ x ∈ p, f x ∈ q) (x : p) : f.restrict hf x = ⟨f x, hf x.1 x.2⟩ :=
  rfl


lemma restrict_sub {R M M₁ : Type*}
    [Ring R] [AddCommGroup M] [AddCommGroup M₁] [Module R M] [Module R M₁]
    {p : Submodule R M} {q : Submodule R M₁} {f g : M →ₗ[R] M₁}
    (hf : MapsTo f p q) (hg : MapsTo g p q)
    (hfg : MapsTo (f - g) p q := fun _ hx ↦ q.sub_mem (hf hx) (hg hx)) :
    f.restrict hf - g.restrict hg = (f - g).restrict hfg := by
  /-
    R : Type u_10
    M : Type u_11
    M₁ : Type u_12
    inst✝⁴ : Ring R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup M₁
    inst✝¹ : Module R M
    inst✝ : Module R M₁
    p : Submodule R M
    q : Submodule R M₁
    f g : LinearMap (RingHom.id R) M M₁
    hf : Set.MapsTo ⇑f ↑p ↑q
    hg : Set.MapsTo ⇑g ↑p ↑q
    hfg : optParam (Set.MapsTo ⇑(HSub.hSub f g) ↑p ↑q) ⋯
    ⊢ Eq (HSub.hSub (f.restrict hf) (g.restrict hg)) ((HSub.hSub f g).restrict hfg)
  -/
  ext; simp
       /-
         🎉 no goals
       -/


lemma restrict_comp
    {M₂ M₃ : Type*} [AddCommMonoid M₂] [AddCommMonoid M₃] [Module R M₂] [Module R M₃]
    {p : Submodule R M} {p₂ : Submodule R M₂} {p₃ : Submodule R M₃}
    {f : M →ₗ[R] M₂} {g : M₂ →ₗ[R] M₃}
    (hf : MapsTo f p p₂) (hg : MapsTo g p₂ p₃) (hfg : MapsTo (g ∘ₗ f) p p₃ := hg.comp hf) :
    (g ∘ₗ f).restrict hfg = (g.restrict hg) ∘ₗ (f.restrict hf) :=
  rfl

-- TODO Consider defining `Algebra R (p.compatibleMaps p)`, `AlgHom` version of `LinearMap.restrict`

lemma restrict_smul_one
    {R M : Type*} [CommSemiring R] [AddCommMonoid M] [Module R M] {p : Submodule R M}
    (μ : R) (h : ∀ x ∈ p, (μ • (1 : Module.End R M)) x ∈ p := fun _ ↦ p.smul_mem μ) :
    (μ • 1 : Module.End R M).restrict h = μ • (1 : Module.End R p) :=
  rfl


lemma restrict_commute {f g : M →ₗ[R] M} (h : Commute f g) {p : Submodule R M}
    (hf : MapsTo f p p) (hg : MapsTo g p p) :
    Commute (f.restrict hf) (g.restrict hg) := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) M M
    h : Commute f g
    p : Submodule R M
    hf : Set.MapsTo ⇑f ↑p ↑p
    hg : Set.MapsTo ⇑g ↑p ↑p
    ⊢ Commute (f.restrict hf) (g.restrict hg)
  -/
  change _ * _ = _ * _
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) M M
    h : Commute f g
    p : Submodule R M
    hf : Set.MapsTo ⇑f ↑p ↑p
    hg : Set.MapsTo ⇑g ↑p ↑p
    ⊢ Eq (HMul.hMul (f.restrict hf) (g.restrict hg)) (HMul.hMul (g.restrict hg) (f …
  -/
  conv_lhs => rw [mul_eq_comp, ← restrict_comp]; congr; rw [← mul_eq_comp, h.eq]
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f g : LinearMap (RingHom.id R) M M
    h : Commute f g
    p : Submodule R M
    hf : Set.MapsTo ⇑f ↑p ↑p
    hg : Set.MapsTo ⇑g ↑p ↑p
    ⊢ Eq ((HMul.hMul g f).restrict ⋯) (HMul.hMul (g.restrict hg) (f.restrict hf))
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem subtype_comp_restrict {f : M →ₗ[R] M₁} {p : Submodule R M} {q : Submodule R M₁}
    (hf : ∀ x ∈ p, f x ∈ q) : q.subtype.comp (f.restrict hf) = f.domRestrict p :=
  rfl


theorem restrict_eq_codRestrict_domRestrict {f : M →ₗ[R] M₁} {p : Submodule R M}
    {q : Submodule R M₁} (hf : ∀ x ∈ p, f x ∈ q) :
    f.restrict hf = (f.domRestrict p).codRestrict q fun x => hf x.1 x.2 :=
  rfl


theorem restrict_eq_domRestrict_codRestrict {f : M →ₗ[R] M₁} {p : Submodule R M}
    {q : Submodule R M₁} (hf : ∀ x, f x ∈ q) :
    (f.restrict fun x _ => hf x) = (f.codRestrict q hf).domRestrict p :=
  rfl


theorem sum_apply (t : Finset ι) (f : ι → M →ₛₗ[σ₁₂] M₂) (b : M) :
    (∑ d ∈ t, f d) b = ∑ d ∈ t, f d b :=
  _root_.map_sum ((AddMonoidHom.eval b).comp toAddMonoidHom') f _


@[simp, norm_cast]
theorem coeFn_sum {ι : Type*} (t : Finset ι) (f : ι → M →ₛₗ[σ₁₂] M₂) :
    ⇑(∑ i ∈ t, f i) = ∑ i ∈ t, (f i : M → M₂) :=
  _root_.map_sum
    (show AddMonoidHom (M →ₛₗ[σ₁₂] M₂) (M → M₂)
      from { toFun := DFunLike.coe,
             map_zero' := rfl
             map_add' := fun _ _ => rfl }) _ _


theorem submodule_pow_eq_zero_of_pow_eq_zero {N : Submodule R M} {g : Module.End R N}
    {G : Module.End R M} (h : G.comp N.subtype = N.subtype.comp g) {k : ℕ} (hG : G ^ k = 0) :
    g ^ k = 0 := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    g : Module.End R (Subtype fun x => Membership.mem N x)
    G : Module.End R M
    h : Eq (LinearMap.comp G N.subtype) (N.subtype.comp g)
    k : Nat
    hG : Eq (HPow.hPow G k) 0
    ⊢ Eq (HPow.hPow g k) 0
  -/
  ext m
  have hg : N.subtype.comp (g ^ k) m = 0 := by
    rw [← commute_pow_left_of_commute h, hG, zero_comp, zero_apply]
  /-
    case h.a
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    N : Submodule R M
    g : Module.End R (Subtype fun x => Membership.mem N x)
    G : Module.End R M
    h : Eq (LinearMap.comp G N.subtype) (N.subtype.comp g)
    k : Nat
    hG : Eq (HPow.hPow G k) 0
    m : Subtype fun x => Membership.mem N x
    hg : Eq ((N.subtype.comp (HPow.hPow g k)) m) 0
    ⊢ Eq ↑((HPow.hPow g k) m) ↑(0 m)
  -/
  simpa using hg
  /-
    🎉 no goals
  -/


theorem pow_apply_mem_of_forall_mem {p : Submodule R M} (n : ℕ) (h : ∀ x ∈ p, f' x ∈ p) (x : M)
    (hx : x ∈ p) : (f' ^ n) x ∈ p := by
  induction n generalizing x with
  | zero => simpa
  | succ n ih =>
    simpa only [iterate_succ, coe_comp, Function.comp_apply, restrict_apply] using ih _ (h _ hx)


theorem pow_restrict {p : Submodule R M} (n : ℕ) (h : ∀ x ∈ p, f' x ∈ p)
    (h' := pow_apply_mem_of_forall_mem n h) :
    (f'.restrict h) ^ n = (f' ^ n).restrict h' := by
  /-
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    p : Submodule R M
    n : Nat
    h : ∀ (x : M), Membership.mem p x → Membership.mem p (f' x)
    h' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p ((HPow.hPow f' …
    ⊢ Eq (HPow.hPow (f'.restrict h) n) ((HPow.hPow f' n).restrict h')
  -/
  ext x
  /-
    case h.a
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    p : Submodule R M
    n : Nat
    h : ∀ (x : M), Membership.mem p x → Membership.mem p (f' x)
    h' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p ((HPow.hPow f' …
    x : Subtype fun x => Membership.mem p x
    ⊢ Eq ↑((HPow.hPow (f'.restrict h) n) x) ↑(((HPow.hPow f' n).restrict h') x)
  -/
  have : Semiconj (↑) (f'.restrict h) f' := fun _ ↦ restrict_coe_apply _ _ _
  /-
    case h.a
    R : Type u_1
    M : Type u_5
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    f' : LinearMap (RingHom.id R) M M
    p : Submodule R M
    n : Nat
    h : ∀ (x : M), Membership.mem p x → Membership.mem p (f' x)
    h' : optParam (∀ (x : M), Membership.mem p x → Membership.mem p ((HPow.hPow f' …
    x : Subtype fun x => Membership.mem p x
    this : Function.Semiconj Subtype.val ⇑(f'.restrict h) ⇑f'
    ⊢ Eq ↑((HPow.hPow (f'.restrict h) n) x) ↑(((HPow.hPow f' n).restrict h') x)
  -/
  simp [coe_pow, this.iterate_right _ _]
  /-
    🎉 no goals
  -/


/-- Alternative version of `domRestrict` as a linear map. -/
def domRestrict' (p : Submodule R M) : (M →ₗ[R] M₂) →ₗ[R] p →ₗ[R] M₂ where
  toFun φ := φ.domRestrict p
                 /-
                   R : Type u_1
                   R₁ : Type u_2
                   R₂ : Type u_3
                   R₃ : Type u_4
                   M : Type u_5
                   M₁ : Type u_6
                   M₂ : Type u_7
                   M₃ : Type u_8
                   ι : Type u_9
                   inst✝⁴ : CommSemiring R
                   inst✝³ : AddCommMonoid M
                   inst✝² : AddCommMonoid M₂
                   inst✝¹ : Module R M
                   inst✝ : Module R M₂
                   f g : LinearMap (RingHom.id R) M M₂
                   p : Submodule R M
                   ⊢ ∀ (x y : LinearMap (RingHom.id R) M M₂), Eq ((fun φ => φ.domRestrict p) (HAd …
                 -/
  map_add' := by simp [LinearMap.ext_iff]
                 /-
                   🎉 no goals
                 -/
                  /-
                    R : Type u_1
                    R₁ : Type u_2
                    R₂ : Type u_3
                    R₃ : Type u_4
                    M : Type u_5
                    M₁ : Type u_6
                    M₂ : Type u_7
                    M₃ : Type u_8
                    ι : Type u_9
                    inst✝⁴ : CommSemiring R
                    inst✝³ : AddCommMonoid M
                    inst✝² : AddCommMonoid M₂
                    inst✝¹ : Module R M
                    inst✝ : Module R M₂
                    f g : LinearMap (RingHom.id R) M M₂
                    p : Submodule R M
                    ⊢ ∀ (m : R) (x : LinearMap (RingHom.id R) M M₂), Eq ({ toFun := fun φ => φ.dom …
                  -/
  map_smul' := by simp [LinearMap.ext_iff]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem domRestrict'_apply (f : M →ₗ[R] M₂) (p : Submodule R M) (x : p) :
    domRestrict' p f x = f x :=
  rfl


/-- If two submodules `p` and `p'` satisfy `p ⊆ p'`, then `inclusion p p'` is the linear map version
of this inclusion. -/
def inclusion (h : p ≤ p') : p →ₗ[R] p' :=
  p.subtype.codRestrict p' fun ⟨_, hx⟩ => h hx


@[simp]
theorem coe_inclusion (h : p ≤ p') (x : p) : (inclusion h x : M) = x :=
  rfl


theorem inclusion_apply (h : p ≤ p') (x : p) : inclusion h x = ⟨x, h x.2⟩ :=
  rfl


theorem inclusion_injective (h : p ≤ p') : Function.Injective (inclusion h) := fun _ _ h =>
  Subtype.val_injective (Subtype.mk.inj h)


theorem subtype_comp_inclusion (p q : Submodule R M) (h : p ≤ q) :
    q.subtype.comp (inclusion h) = p.subtype := by
  /-
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p q : Submodule R M
    h : LE.le p q
    ⊢ Eq (q.subtype.comp (Submodule.inclusion h)) p.subtype
  -/
  ext ⟨b, hb⟩
  /-
    case h.mk
    R : Type u_1
    M : Type u_2
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    p q : Submodule R M
    h : LE.le p q
    b : M
    hb : Membership.mem p b
    ⊢ Eq ((q.subtype.comp (Submodule.inclusion h)) ⟨b, hb⟩) (p.subtype ⟨b, hb⟩)
  -/
  rfl
  /-
    🎉 no goals
  -/


