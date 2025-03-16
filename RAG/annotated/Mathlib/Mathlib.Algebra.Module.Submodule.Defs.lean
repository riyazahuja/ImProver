/-- A submodule of a module is one which is closed under vector operations.
  This is a sufficient condition for the subset of vectors in the submodule
  to themselves form a module. -/
structure Submodule (R : Type u) (M : Type v) [Semiring R] [AddCommMonoid M] [Module R M] extends
  AddSubmonoid M, SubMulAction R M : Type v


instance setLike : SetLike (Submodule R M) M where
  coe s := s.carrier
                             /-
                               G : Type u''
                               S : Type u'
                               R : Type u
                               M : Type v
                               ι : Type w
                               inst✝² : Semiring R
                               inst✝¹ : AddCommMonoid M
                               inst✝ : Module R M
                               p q : Submodule R M
                               h : Eq ((fun s => s.carrier) p) ((fun s => s.carrier) q)
                               ⊢ Eq p q
                             -/
  coe_injective' p q h := by cases p; cases q; congr; exact SetLike.coe_injective' h
                                                      /-
                                                        🎉 no goals
                                                      -/


instance addSubmonoidClass : AddSubmonoidClass (Submodule R M) M where
  zero_mem _ := AddSubmonoid.zero_mem' _
  add_mem := AddSubsemigroup.add_mem' _


instance smulMemClass : SMulMemClass (Submodule R M) R M where
  smul_mem {s} c _ h := SubMulAction.smul_mem' s.toSubMulAction c h


@[simp]
theorem mem_toAddSubmonoid (p : Submodule R M) (x : M) : x ∈ p.toAddSubmonoid ↔ x ∈ p :=
  Iff.rfl


@[simp]
theorem mem_mk {S : AddSubmonoid M} {x : M} (h) : x ∈ (⟨S, h⟩ : Submodule R M) ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem coe_set_mk (S : AddSubmonoid M) (h) : ((⟨S, h⟩ : Submodule R M) : Set M) = S :=
  rfl


@[simp] theorem eta (h) : ({p with smul_mem' := h} : Submodule R M) = p :=
  rfl

-- Porting note: replaced `S ⊆ S' : Set` with `S ≤ S'`

@[simp]
theorem mk_le_mk {S S' : AddSubmonoid M} (h h') :
    (⟨S, h⟩ : Submodule R M) ≤ (⟨S', h'⟩ : Submodule R M) ↔ S ≤ S' :=
  Iff.rfl


@[ext]
theorem ext (h : ∀ x, x ∈ p ↔ x ∈ q) : p = q :=
  SetLike.ext h

-- Porting note: adding this as the `simp`-normal form of `toSubMulAction_inj`

@[simp]
theorem carrier_inj : p.carrier = q.carrier ↔ p = q :=
  (SetLike.coe_injective (A := Submodule R M)).eq_iff


/-- Copy of a submodule with a new `carrier` equal to the old one. Useful to fix definitional
equalities. -/
protected def copy (p : Submodule R M) (s : Set M) (hs : s = ↑p) : Submodule R M where
  carrier := s
                  /-
                    G : Type u''
                    S : Type u'
                    R : Type u
                    M : Type v
                    ι : Type w
                    inst✝² : Semiring R
                    inst✝¹ : AddCommMonoid M
                    inst✝ : Module R M
                    p✝ q p : Submodule R M
                    s : Set M
                    hs : Eq s ↑p
                    ⊢ Membership.mem { carrier := s, add_mem' := ⋯ }.carrier 0
                  -/
  zero_mem' := by simpa [hs] using p.zero_mem'
                  /-
                    🎉 no goals
                  -/
  add_mem' := hs.symm ▸ p.add_mem'
                  /-
                    G : Type u''
                    S : Type u'
                    R : Type u
                    M : Type v
                    ι : Type w
                    inst✝² : Semiring R
                    inst✝¹ : AddCommMonoid M
                    inst✝ : Module R M
                    p✝ q p : Submodule R M
                    s : Set M
                    hs : Eq s ↑p
                    ⊢ ∀ (c : R) {x : M}, Membership.mem { carrier := s, add_mem' := ⋯, zero_mem' : …
                  -/
  smul_mem' := by simpa [hs] using p.smul_mem'
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem coe_copy (S : Submodule R M) (s : Set M) (hs : s = ↑S) : (S.copy s hs : Set M) = s :=
  rfl


theorem copy_eq (S : Submodule R M) (s : Set M) (hs : s = ↑S) : S.copy s hs = S :=
  SetLike.coe_injective hs


theorem toAddSubmonoid_injective : Injective (toAddSubmonoid : Submodule R M → AddSubmonoid M) :=
  fun p q h => SetLike.ext'_iff.2 (show (p.toAddSubmonoid : Set M) = q from SetLike.ext'_iff.1 h)


@[simp]
theorem toAddSubmonoid_inj : p.toAddSubmonoid = q.toAddSubmonoid ↔ p = q :=
  toAddSubmonoid_injective.eq_iff


@[deprecated (since := "2024-12-29")] alias toAddSubmonoid_eq := toAddSubmonoid_inj


@[simp]
theorem coe_toAddSubmonoid (p : Submodule R M) : (p.toAddSubmonoid : Set M) = p :=
  rfl


theorem toSubMulAction_injective : Injective (toSubMulAction : Submodule R M → SubMulAction R M) :=
  fun p q h => SetLike.ext'_iff.2 (show (p.toSubMulAction : Set M) = q from SetLike.ext'_iff.1 h)


theorem toSubMulAction_inj : p.toSubMulAction = q.toSubMulAction ↔ p = q :=
  toSubMulAction_injective.eq_iff


@[deprecated (since := "2024-12-29")] alias toSubMulAction_eq := toSubMulAction_inj


@[simp]
theorem coe_toSubMulAction (p : Submodule R M) : (p.toSubMulAction : Set M) = p :=
  rfl


/-- A submodule of a `Module` is a `Module`. -/
instance (priority := 75) toModule : Module R S' :=
  Subtype.coe_injective.module R (AddSubmonoidClass.subtype S') (SetLike.val_smul S')


/-- This can't be an instance because Lean wouldn't know how to find `R`, but we can still use
this to manually derive `Module` on specific types. -/
def toModule' (S R' R A : Type*) [Semiring R] [NonUnitalNonAssocSemiring A]
    [Module R A] [Semiring R'] [SMul R' R] [Module R' A] [IsScalarTower R' R A]
    [SetLike S A] [AddSubmonoidClass S A] [SMulMemClass S R A] (s : S) :
    Module R' s :=
  haveI : SMulMemClass S R' A := SMulMemClass.ofIsScalarTower S R' R A
  SMulMemClass.toModule s


theorem mem_carrier : x ∈ p.carrier ↔ x ∈ (p : Set M) :=
  Iff.rfl


@[simp]
protected theorem zero_mem : (0 : M) ∈ p :=
  zero_mem _


protected theorem add_mem (h₁ : x ∈ p) (h₂ : y ∈ p) : x + y ∈ p :=
  add_mem h₁ h₂


theorem smul_mem (r : R) (h : x ∈ p) : r • x ∈ p :=
  p.smul_mem' r h


theorem smul_of_tower_mem [SMul S R] [SMul S M] [IsScalarTower S R M] (r : S) (h : x ∈ p) :
    r • x ∈ p :=
  p.toSubMulAction.smul_of_tower_mem r h


@[simp]
theorem smul_mem_iff' [Group G] [MulAction G M] [SMul G R] [IsScalarTower G R M] (g : G) :
    g • x ∈ p ↔ x ∈ p :=
  p.toSubMulAction.smul_mem_iff' g


instance add : Add p :=
  ⟨fun x y => ⟨x.1 + y.1, add_mem x.2 y.2⟩⟩


instance zero : Zero p :=
  ⟨⟨0, zero_mem _⟩⟩


instance inhabited : Inhabited p :=
  ⟨0⟩


instance smul [SMul S R] [SMul S M] [IsScalarTower S R M] : SMul S p :=
  ⟨fun c x => ⟨c • x.1, smul_of_tower_mem _ c x.2⟩⟩


instance isScalarTower [SMul S R] [SMul S M] [IsScalarTower S R M] : IsScalarTower S R p :=
  p.toSubMulAction.isScalarTower


instance isScalarTower' {S' : Type*} [SMul S R] [SMul S M] [SMul S' R] [SMul S' M] [SMul S S']
    [IsScalarTower S' R M] [IsScalarTower S S' M] [IsScalarTower S R M] : IsScalarTower S S' p :=
  p.toSubMulAction.isScalarTower'


protected theorem nonempty : (p : Set M).Nonempty :=
  ⟨0, p.zero_mem⟩


@[simp]
theorem mk_eq_zero {x} (h : x ∈ p) : (⟨x, h⟩ : p) = 0 ↔ x = 0 :=
  Subtype.ext_iff_val


@[norm_cast] -- Porting note: removed `@[simp]` because this follows from `ZeroMemClass.coe_zero`
theorem coe_eq_zero {x : p} : (x : M) = 0 ↔ x = 0 :=
  (SetLike.coe_eq_coe : (x : M) = (0 : p) ↔ x = 0)


@[simp, norm_cast]
theorem coe_add (x y : p) : (↑(x + y) : M) = ↑x + ↑y :=
  rfl


@[simp, norm_cast]
theorem coe_zero : ((0 : p) : M) = 0 :=
  rfl


@[norm_cast]
theorem coe_smul (r : R) (x : p) : ((r • x : p) : M) = r • (x : M) :=
  rfl


@[simp, norm_cast]
theorem coe_smul_of_tower [SMul S R] [SMul S M] [IsScalarTower S R M] (r : S) (x : p) :
    ((r • x : p) : M) = r • (x : M) :=
  rfl


@[norm_cast] -- Porting note: removed `@[simp]` because this is now structure eta
theorem coe_mk (x : M) (hx : x ∈ p) : ((⟨x, hx⟩ : p) : M) = x :=
  rfl

-- Porting note: removed `@[simp]` because this is exactly `SetLike.coe_mem`

theorem coe_mem (x : p) : (x : M) ∈ p :=
  x.2


instance addCommMonoid : AddCommMonoid p :=
  { p.toAddSubmonoid.toAddCommMonoid with }


instance module' [Semiring S] [SMul S R] [Module S M] [IsScalarTower S R M] : Module S p :=
  { (show MulAction S p from p.toSubMulAction.mulAction') with
                             /-
                               G : Type u''
                               S : Type u'
                               R : Type u
                               M : Type v
                               ι : Type w
                               inst✝⁵ : Semiring R
                               inst✝⁴ : AddCommMonoid M
                               module_M : Module R M
                               p q : Submodule R M
                               r : R
                               x y : M
                               inst✝³ : Semiring S
                               inst✝² : SMul S R
                               inst✝¹ : Module S M
                               inst✝ : IsScalarTower S R M
                               a : S
                               ⊢ Eq (HSMul.hSMul a 0) 0
                             -/
    smul_zero := fun a => by ext; simp
                                  /-
                                    🎉 no goals
                                  -/
                             /-
                               G : Type u''
                               S : Type u'
                               R : Type u
                               M : Type v
                               ι : Type w
                               inst✝⁵ : Semiring R
                               inst✝⁴ : AddCommMonoid M
                               module_M : Module R M
                               p q : Submodule R M
                               r : R
                               x y : M
                               inst✝³ : Semiring S
                               inst✝² : SMul S R
                               inst✝¹ : Module S M
                               inst✝ : IsScalarTower S R M
                               a : Subtype fun x => Membership.mem p x
                               ⊢ Eq (HSMul.hSMul 0 a) 0
                             -/
                                /-
                                  G : Type u''
                                  S : Type u'
                                  R : Type u
                                  M : Type v
                                  ι : Type w
                                  inst✝⁵ : Semiring R
                                  inst✝⁴ : AddCommMonoid M
                                  module_M : Module R M
                                  p q : Submodule R M
                                  r : R
                                  x✝ y : M
                                  inst✝³ : Semiring S
                                  inst✝² : SMul S R
                                  inst✝¹ : Module S M
                                  inst✝ : IsScalarTower S R M
                                  a b : S
                                  x : Subtype fun x => Membership.mem p x
                                  ⊢ Eq (HSMul.hSMul (HAdd.hAdd a b) x) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
                                -/
                                /-
                                  G : Type u''
                                  S : Type u'
                                  R : Type u
                                  M : Type v
                                  ι : Type w
                                  inst✝⁵ : Semiring R
                                  inst✝⁴ : AddCommMonoid M
                                  module_M : Module R M
                                  p q : Submodule R M
                                  r : R
                                  x✝ y✝ : M
                                  inst✝³ : Semiring S
                                  inst✝² : SMul S R
                                  inst✝¹ : Module S M
                                  inst✝ : IsScalarTower S R M
                                  a : S
                                  x y : Subtype fun x => Membership.mem p x
                                  ⊢ Eq (HSMul.hSMul a (HAdd.hAdd x y)) (HAdd.hAdd (HSMul.hSMul a x) (HSMul.hSMul …
                                -/
    zero_smul := fun a => by ext; simp
                                     /-
                                       🎉 no goals
                                     -/
                                     /-
                                       🎉 no goals
                                     -/
                                  /-
                                    🎉 no goals
                                  -/
    add_smul := fun a b x => by ext; simp [add_smul]
    smul_add := fun a x y => by ext; simp [smul_add] }


instance module : Module R p :=
  p.module'


instance addSubgroupClass [Module R M] : AddSubgroupClass (Submodule R M) M :=
  { Submodule.addSubmonoidClass with neg_mem := fun p {_} => p.toSubMulAction.neg_mem }


protected theorem neg_mem (hx : x ∈ p) : -x ∈ p :=
  neg_mem hx


/-- Reinterpret a submodule as an additive subgroup. -/
def toAddSubgroup : AddSubgroup M :=
  { p.toAddSubmonoid with neg_mem' := fun {_} => p.neg_mem }


@[simp]
theorem coe_toAddSubgroup : (p.toAddSubgroup : Set M) = p :=
  rfl


@[simp]
theorem mem_toAddSubgroup : x ∈ p.toAddSubgroup ↔ x ∈ p :=
  Iff.rfl


theorem toAddSubgroup_injective : Injective (toAddSubgroup : Submodule R M → AddSubgroup M)
  | _, _, h => SetLike.ext (SetLike.ext_iff.1 h : _)


@[simp]
theorem toAddSubgroup_inj : p.toAddSubgroup = p'.toAddSubgroup ↔ p = p' :=
  toAddSubgroup_injective.eq_iff


@[deprecated (since := "2024-12-29")] alias toAddSubgroup_eq := toAddSubgroup_inj


protected theorem sub_mem : x ∈ p → y ∈ p → x - y ∈ p :=
  sub_mem


protected theorem neg_mem_iff : -x ∈ p ↔ x ∈ p :=
  neg_mem_iff


protected theorem add_mem_iff_left : y ∈ p → (x + y ∈ p ↔ x ∈ p) :=
  add_mem_cancel_right


protected theorem add_mem_iff_right : x ∈ p → (x + y ∈ p ↔ y ∈ p) :=
  add_mem_cancel_left


protected theorem coe_neg (x : p) : ((-x : p) : M) = -x :=
  NegMemClass.coe_neg _


protected theorem coe_sub (x y : p) : (↑(x - y) : M) = ↑x - ↑y :=
  AddSubgroupClass.coe_sub _ _


theorem sub_mem_iff_left (hy : y ∈ p) : x - y ∈ p ↔ x ∈ p := by
  /-
    R : Type u
    M : Type v
    inst✝¹ : Ring R
    inst✝ : AddCommGroup M
    module_M : Module R M
    p : Submodule R M
    x y : M
    hy : Membership.mem p y
    ⊢ Iff (Membership.mem p (HSub.hSub x y)) (Membership.mem p x)
  -/
  rw [sub_eq_add_neg, p.add_mem_iff_left (p.neg_mem hy)]
  /-
    🎉 no goals
  -/


theorem sub_mem_iff_right (hx : x ∈ p) : x - y ∈ p ↔ y ∈ p := by
  /-
    R : Type u
    M : Type v
    inst✝¹ : Ring R
    inst✝ : AddCommGroup M
    module_M : Module R M
    p : Submodule R M
    x y : M
    hx : Membership.mem p x
    ⊢ Iff (Membership.mem p (HSub.hSub x y)) (Membership.mem p y)
  -/
  rw [sub_eq_add_neg, p.add_mem_iff_right hx, p.neg_mem_iff]
  /-
    🎉 no goals
  -/


instance addCommGroup : AddCommGroup p :=
  { p.toAddSubgroup.toAddCommGroup with }


instance (priority := 75) module' {T : Type*} [Semiring R] [AddCommMonoid M] [Semiring S]
    [Module R M] [SMul S R] [Module S M] [IsScalarTower S R M] [SetLike T M] [AddSubmonoidClass T M]
    [SMulMemClass T R M] (t : T) : Module S t where
                   /-
                     G : Type u''
                     S : Type u'
                     R : Type u
                     M : Type v
                     ι : Type w
                     T : Type u_1
                     inst✝⁹ : Semiring R
                     inst✝⁸ : AddCommMonoid M
                     inst✝⁷ : Semiring S
                     inst✝⁶ : Module R M
                     inst✝⁵ : SMul S R
                     inst✝⁴ : Module S M
                     inst✝³ : IsScalarTower S R M
                     inst✝² : SetLike T M
                     inst✝¹ : AddSubmonoidClass T M
                     inst✝ : SMulMemClass T R M
                     t : T
                     x✝ : Subtype fun x => Membership.mem t x
                     ⊢ Eq (HSMul.hSMul 1 x✝) x✝
                   -/
  one_smul _ := by ext; simp
                        /-
                          🎉 no goals
                        -/
                       /-
                         G : Type u''
                         S : Type u'
                         R : Type u
                         M : Type v
                         ι : Type w
                         T : Type u_1
                         inst✝⁹ : Semiring R
                         inst✝⁸ : AddCommMonoid M
                         inst✝⁷ : Semiring S
                         inst✝⁶ : Module R M
                         inst✝⁵ : SMul S R
                         inst✝⁴ : Module S M
                         inst✝³ : IsScalarTower S R M
                         inst✝² : SetLike T M
                         inst✝¹ : AddSubmonoidClass T M
                         inst✝ : SMulMemClass T R M
                         t : T
                         x✝² x✝¹ : S
                         x✝ : Subtype fun x => Membership.mem t x
                         ⊢ Eq (HSMul.hSMul (HMul.hMul x✝² x✝¹) x✝) (HSMul.hSMul x✝² (HSMul.hSMul x✝¹ x✝))
                       -/
  mul_smul _ _ _ := by ext; simp [mul_smul]
                            /-
                              🎉 no goals
                            -/
                    /-
                      G : Type u''
                      S : Type u'
                      R : Type u
                      M : Type v
                      ι : Type w
                      T : Type u_1
                      inst✝⁹ : Semiring R
                      inst✝⁸ : AddCommMonoid M
                      inst✝⁷ : Semiring S
                      inst✝⁶ : Module R M
                      inst✝⁵ : SMul S R
                      inst✝⁴ : Module S M
                      inst✝³ : IsScalarTower S R M
                      inst✝² : SetLike T M
                      inst✝¹ : AddSubmonoidClass T M
                      inst✝ : SMulMemClass T R M
                      t : T
                      x✝ : S
                      ⊢ Eq (HSMul.hSMul x✝ 0) 0
                    -/
  smul_zero _ := by ext; simp
                         /-
                           🎉 no goals
                         -/
                    /-
                      G : Type u''
                      S : Type u'
                      R : Type u
                      M : Type v
                      ι : Type w
                      T : Type u_1
                      inst✝⁹ : Semiring R
                      inst✝⁸ : AddCommMonoid M
                      inst✝⁷ : Semiring S
                      inst✝⁶ : Module R M
                      inst✝⁵ : SMul S R
                      inst✝⁴ : Module S M
                      inst✝³ : IsScalarTower S R M
                      inst✝² : SetLike T M
                      inst✝¹ : AddSubmonoidClass T M
                      inst✝ : SMulMemClass T R M
                      t : T
                      x✝ : Subtype fun x => Membership.mem t x
                      ⊢ Eq (HSMul.hSMul 0 x✝) 0
                    -/
                       /-
                         G : Type u''
                         S : Type u'
                         R : Type u
                         M : Type v
                         ι : Type w
                         T : Type u_1
                         inst✝⁹ : Semiring R
                         inst✝⁸ : AddCommMonoid M
                         inst✝⁷ : Semiring S
                         inst✝⁶ : Module R M
                         inst✝⁵ : SMul S R
                         inst✝⁴ : Module S M
                         inst✝³ : IsScalarTower S R M
                         inst✝² : SetLike T M
                         inst✝¹ : AddSubmonoidClass T M
                         inst✝ : SMulMemClass T R M
                         t : T
                         x✝² x✝¹ : S
                         x✝ : Subtype fun x => Membership.mem t x
                         ⊢ Eq (HSMul.hSMul (HAdd.hAdd x✝² x✝¹) x✝) (HAdd.hAdd (HSMul.hSMul x✝² x✝) (HSM …
                       -/
                       /-
                         G : Type u''
                         S : Type u'
                         R : Type u
                         M : Type v
                         ι : Type w
                         T : Type u_1
                         inst✝⁹ : Semiring R
                         inst✝⁸ : AddCommMonoid M
                         inst✝⁷ : Semiring S
                         inst✝⁶ : Module R M
                         inst✝⁵ : SMul S R
                         inst✝⁴ : Module S M
                         inst✝³ : IsScalarTower S R M
                         inst✝² : SetLike T M
                         inst✝¹ : AddSubmonoidClass T M
                         inst✝ : SMulMemClass T R M
                         t : T
                         x✝² : S
                         x✝¹ x✝ : Subtype fun x => Membership.mem t x
                         ⊢ Eq (HSMul.hSMul x✝² (HAdd.hAdd x✝¹ x✝)) (HAdd.hAdd (HSMul.hSMul x✝² x✝¹) (HS …
                       -/
  zero_smul _ := by ext; simp
                            /-
                              🎉 no goals
                            -/
                            /-
                              🎉 no goals
                            -/
                         /-
                           🎉 no goals
                         -/
  add_smul _ _ _ := by ext; simp [add_smul]
  smul_add _ _ _ := by ext; simp [smul_add]


instance (priority := 75) module [Semiring R] [AddCommMonoid M] [Module R M] [SetLike S M]
    [AddSubmonoidClass S M] [SMulMemClass S R M] (s : S) : Module R s :=
  module' s


