/-- `M` acts pretransitively on `α` if for any `x y` there is `g` such that `g +ᵥ x = y`.
  A transitive action should furthermore have `α` nonempty. -/
class AddAction.IsPretransitive (M α : Type*) [VAdd M α] : Prop where
  /-- There is `g` such that `g +ᵥ x = y`. -/
  exists_vadd_eq : ∀ x y : α, ∃ g : M, g +ᵥ x = y


/-- `M` acts pretransitively on `α` if for any `x y` there is `g` such that `g • x = y`.
  A transitive action should furthermore have `α` nonempty. -/
@[to_additive]
class MulAction.IsPretransitive (M α : Type*) [SMul M α] : Prop where
  /-- There is `g` such that `g • x = y`. -/
  exists_smul_eq : ∀ x y : α, ∃ g : M, g • x = y


@[to_additive]
lemma exists_smul_eq (x y : α) : ∃ m : M, m • x = y := IsPretransitive.exists_smul_eq x y


@[to_additive]
lemma surjective_smul (x : α) : Surjective fun c : M ↦ c • x := exists_smul_eq M x


/-- The regular action of a group on itself is transitive. -/
@[to_additive "The regular action of a group on itself is transitive."]
instance Regular.isPretransitive [Group G] : IsPretransitive G G :=
  ⟨fun x y ↦ ⟨y * x⁻¹, inv_mul_cancel_right _ _⟩⟩


/-- If an action is transitive, then composing this action with a surjective homomorphism gives
again a transitive action. -/
@[to_additive]
lemma isPretransitive_compHom {E F G : Type*} [Monoid E] [Monoid F] [MulAction F G]
    [IsPretransitive F G] {f : E →* F} (hf : Surjective f) :
    letI : MulAction E G := MulAction.compHom _ f
    IsPretransitive E G := by
  /-
    E : Type u_5
    F : Type u_6
    G : Type u_7
    inst✝³ : Monoid E
    inst✝² : Monoid F
    inst✝¹ : MulAction F G
    inst✝ : MulAction.IsPretransitive F G
    f : MonoidHom E F
    hf : Function.Surjective ⇑f
    ⊢ MulAction.IsPretransitive E G
  -/
  let _ : MulAction E G := MulAction.compHom _ f
  /-
    E : Type u_5
    F : Type u_6
    G : Type u_7
    inst✝³ : Monoid E
    inst✝² : Monoid F
    inst✝¹ : MulAction F G
    inst✝ : MulAction.IsPretransitive F G
    f : MonoidHom E F
    hf : Function.Surjective ⇑f
    x✝ : MulAction E G := MulAction.compHom G f
    ⊢ MulAction.IsPretransitive E G
  -/
  refine ⟨fun x y ↦ ?_⟩
  /-
    E : Type u_5
    F : Type u_6
    G : Type u_7
    inst✝³ : Monoid E
    inst✝² : Monoid F
    inst✝¹ : MulAction F G
    inst✝ : MulAction.IsPretransitive F G
    f : MonoidHom E F
    hf : Function.Surjective ⇑f
    x✝ : MulAction E G := MulAction.compHom G f
    x y : G
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) y
  -/
  obtain ⟨m, rfl⟩ : ∃ m : F, m • x = y := exists_smul_eq F x y
  /-
    case intro
    E : Type u_5
    F : Type u_6
    G : Type u_7
    inst✝³ : Monoid E
    inst✝² : Monoid F
    inst✝¹ : MulAction F G
    inst✝ : MulAction.IsPretransitive F G
    f : MonoidHom E F
    hf : Function.Surjective ⇑f
    x✝ : MulAction E G := MulAction.compHom G f
    x : G
    m : F
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) (HSMul.hSMul m x)
  -/
  obtain ⟨e, rfl⟩ : ∃ e, f e = m := hf m
  /-
    case intro.intro
    E : Type u_5
    F : Type u_6
    G : Type u_7
    inst✝³ : Monoid E
    inst✝² : Monoid F
    inst✝¹ : MulAction F G
    inst✝ : MulAction.IsPretransitive F G
    f : MonoidHom E F
    hf : Function.Surjective ⇑f
    x✝ : MulAction E G := MulAction.compHom G f
    x : G
    e : E
    ⊢ Exists fun g => Eq (HSMul.hSMul g x) (HSMul.hSMul (f e) x)
  -/
  exact ⟨e, rfl⟩
  /-
    🎉 no goals
  -/


@[to_additive]
lemma IsPretransitive.of_smul_eq {M N α : Type*} [SMul M α] [SMul N α] [IsPretransitive M α]
    (f : M → N) (hf : ∀ {c : M} {x : α}, f c • x = c • x) : IsPretransitive N α where
  exists_smul_eq x y := (exists_smul_eq x y).elim fun m h ↦ ⟨f m, hf.trans h⟩


@[to_additive]
lemma IsPretransitive.of_compHom {M N α : Type*} [Monoid M] [Monoid N] [MulAction N α]
    (f : M →* N) [h : letI := compHom α f; IsPretransitive M α] : IsPretransitive N α :=
  letI := compHom α f; h.of_smul_eq f rfl


@[to_additive]
lemma MulAction.IsPretransitive.of_isScalarTower (M : Type*) {N α : Type*} [Monoid N] [SMul M N]
    [MulAction N α] [SMul M α] [IsScalarTower M N α] [IsPretransitive M α] : IsPretransitive N α :=
  of_smul_eq (fun x : M ↦ x • 1) (smul_one_smul N _ _)


instance Additive.addAction_isPretransitive [Monoid α] [MulAction α β]
    [MulAction.IsPretransitive α β] : AddAction.IsPretransitive (Additive α) β :=
  ⟨@MulAction.exists_smul_eq α _ _ _⟩


instance Multiplicative.mulAction_isPretransitive [AddMonoid α] [AddAction α β]
    [AddAction.IsPretransitive α β] : MulAction.IsPretransitive (Multiplicative α) β :=
  ⟨@AddAction.exists_vadd_eq α _ _ _⟩


