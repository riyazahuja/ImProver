@[to_additive]
instance SubmonoidClass.instMulArchimedean {M S : Type*} [SetLike S M] [OrderedCommMonoid M]
    [SubmonoidClass S M] [MulArchimedean M] (H : S) : MulArchimedean H := by
  /-
    M : Type u_1
    S : Type u_2
    inst✝³ : SetLike S M
    inst✝² : OrderedCommMonoid M
    inst✝¹ : SubmonoidClass S M
    inst✝ : MulArchimedean M
    H : S
    ⊢ MulArchimedean (Subtype fun x => Membership.mem H x)
  -/
  constructor
  /-
    case arch
    M : Type u_1
    S : Type u_2
    inst✝³ : SetLike S M
    inst✝² : OrderedCommMonoid M
    inst✝¹ : SubmonoidClass S M
    inst✝ : MulArchimedean M
    H : S
    ⊢ ∀ (x : Subtype fun x => Membership.mem H x) {y : Subtype fun x => Membership …
  -/
  rintro x _
  /-
    case arch
    M : Type u_1
    S : Type u_2
    inst✝³ : SetLike S M
    inst✝² : OrderedCommMonoid M
    inst✝¹ : SubmonoidClass S M
    inst✝ : MulArchimedean M
    H : S
    x y✝ : Subtype fun x => Membership.mem H x
    ⊢ LT.lt 1 y✝ → Exists fun n => LE.le x (HPow.hPow y✝ n)
  -/
  simp only [← Subtype.coe_lt_coe, OneMemClass.coe_one, SubmonoidClass.mk_pow, Subtype.mk_le_mk]
  /-
    case arch
    M : Type u_1
    S : Type u_2
    inst✝³ : SetLike S M
    inst✝² : OrderedCommMonoid M
    inst✝¹ : SubmonoidClass S M
    inst✝ : MulArchimedean M
    H : S
    x y✝ : Subtype fun x => Membership.mem H x
    ⊢ LT.lt 1 ↑y✝ → Exists fun n => LE.le x (HPow.hPow y✝ n)
  -/
  exact MulArchimedean.arch x.val
  /-
    🎉 no goals
  -/

