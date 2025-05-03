/-- `Bundle.TotalSpace F E` is the total space of the bundle. It consists of pairs
`(proj : B, snd : E proj)`.
-/
@[ext]
structure TotalSpace (F : Type*) (E : B → Type*) where
  /-- `Bundle.TotalSpace.proj` is the canonical projection `Bundle.TotalSpace F E → B` from the
  total space to the base space. -/
  proj : B
  snd : E proj


instance [Inhabited B] [Inhabited (E default)] : Inhabited (TotalSpace F E) :=
  ⟨⟨default, default⟩⟩


@[inherit_doc]
scoped notation:max "π" F':max E':max => Bundle.TotalSpace.proj (F := F') (E := E')


abbrev TotalSpace.mk' (F : Type*) (x : B) (y : E x) : TotalSpace F E := ⟨x, y⟩


theorem TotalSpace.mk_cast {x x' : B} (h : x = x') (b : E x) :
                                                                 /-
                                                                   B : Type u_1
                                                                   F : Type u_2
                                                                   E : B → Type u_3
                                                                   x x' : B
                                                                   h : Eq x x'
                                                                   b : E x
                                                                   ⊢ Eq (Bundle.TotalSpace.mk' F x' (cast ⋯ b)) { proj := x, snd := b }
                                                                 -/
    .mk' F x' (cast (congr_arg E h) b) = TotalSpace.mk x b := by subst h; rfl
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp 1001, mfld_simps 1001]
theorem TotalSpace.mk_inj {b : B} {y y' : E b} : mk' F b y = mk' F b y' ↔ y = y' := by
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    b : B
    y y' : E b
    ⊢ Iff (Eq (Bundle.TotalSpace.mk' F b y) (Bundle.TotalSpace.mk' F b y')) (Eq y  …
  -/
  simp [TotalSpace.ext_iff]
  /-
    🎉 no goals
  -/


theorem TotalSpace.mk_injective (b : B) : Injective (mk b : E b → TotalSpace F E) := fun _ _ ↦
  mk_inj.1


instance {x : B} : CoeTC (E x) (TotalSpace F E) :=
  ⟨TotalSpace.mk x⟩


theorem TotalSpace.eta (z : TotalSpace F E) : TotalSpace.mk z.proj z.2 = z := rfl


@[simp]
theorem TotalSpace.exists {p : TotalSpace F E → Prop} : (∃ x, p x) ↔ ∃ b y, p ⟨b, y⟩ :=
  ⟨fun ⟨x, hx⟩ ↦ ⟨x.1, x.2, hx⟩, fun ⟨b, y, h⟩ ↦ ⟨⟨b, y⟩, h⟩⟩


@[simp]
theorem TotalSpace.range_mk (b : B) : range ((↑) : E b → TotalSpace F E) = π F E ⁻¹' {b} := by
  /-
    B : Type u_1
    F : Type u_2
    E : B → Type u_3
    b : B
    ⊢ Eq (Set.range (Bundle.TotalSpace.mk b)) (Set.preimage Bundle.TotalSpace.proj …
  -/
  apply Subset.antisymm
    /-
      case h₁
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      b : B
      ⊢ HasSubset.Subset (Set.range (Bundle.TotalSpace.mk b)) (Set.preimage Bundle.T …
    -/
  · rintro _ ⟨x, rfl⟩
    /-
      case h₁.intro
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      b : B
      x : E b
      ⊢ Membership.mem (Set.preimage Bundle.TotalSpace.proj (Singleton.singleton b)) …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h₂
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      b : B
      ⊢ HasSubset.Subset (Set.preimage Bundle.TotalSpace.proj (Singleton.singleton b …
    -/
  · rintro ⟨_, x⟩ rfl
    /-
      case h₂.mk
      B : Type u_1
      F : Type u_2
      E : B → Type u_3
      proj✝ : B
      x : E proj✝
      ⊢ Membership.mem (Set.range (Bundle.TotalSpace.mk { proj := proj✝, snd := x }. …
    -/
    exact ⟨x, rfl⟩
    /-
      🎉 no goals
    -/


/-- Notation for the direct sum of two bundles over the same base. -/
notation:100 E₁ " ×ᵇ " E₂ => fun x => E₁ x × E₂ x


/-- `Bundle.Trivial B F` is the trivial bundle over `B` of fiber `F`. -/
@[reducible, nolint unusedArguments]
def Trivial (B : Type*) (F : Type*) : B → Type _ := fun _ => F


/-- The trivial bundle, unlike other bundles, has a canonical projection on the fiber. -/
def TotalSpace.trivialSnd (B : Type*) (F : Type*) : TotalSpace F (Bundle.Trivial B F) → F :=
  TotalSpace.snd


/-- A trivial bundle is equivalent to the product `B × F`. -/
@[simps (config := { attrs := [`mfld_simps] })]
def TotalSpace.toProd (B F : Type*) : (TotalSpace F fun _ : B => F) ≃ B × F where
  toFun x := (x.1, x.2)
  invFun x := ⟨x.1, x.2⟩
  left_inv := fun ⟨_, _⟩ => rfl
  right_inv := fun ⟨_, _⟩ => rfl


/-- The pullback of a bundle `E` over a base `B` under a map `f : B' → B`, denoted by
`Bundle.Pullback f E` or `f *ᵖ E`, is the bundle over `B'` whose fiber over `b'` is `E (f b')`. -/
def Pullback (f : B' → B) (E : B → Type*) : B' → Type _ := fun x => E (f x)


@[inherit_doc]
notation f " *ᵖ " E:arg => Pullback f E


instance {f : B' → B} {x : B'} [Nonempty (E (f x))] : Nonempty ((f *ᵖ E) x) :=
  ‹Nonempty (E (f x))›


/-- Natural embedding of the total space of `f *ᵖ E` into `B' × TotalSpace F E`. -/
@[simp]
def pullbackTotalSpaceEmbedding (f : B' → B) : TotalSpace F (f *ᵖ E) → B' × TotalSpace F E :=
  fun z => (z.proj, TotalSpace.mk (f z.proj) z.2)


/-- The base map `f : B' → B` lifts to a canonical map on the total spaces. -/
@[simps (config := { attrs := [`mfld_simps] })]
def Pullback.lift (f : B' → B) : TotalSpace F (f *ᵖ E) → TotalSpace F E := fun z => ⟨f z.proj, z.2⟩


@[simp, mfld_simps]
theorem Pullback.lift_mk (f : B' → B) (x : B') (y : E (f x)) :
    Pullback.lift f (.mk' F x y) = ⟨f x, y⟩ :=
  rfl


