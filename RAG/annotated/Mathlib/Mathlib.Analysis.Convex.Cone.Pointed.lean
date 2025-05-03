local notation3 "𝕜≥0" => {c : 𝕜 // 0 ≤ c}


/-- A pointed cone is a submodule of a module with scalars restricted to being nonnegative. -/
abbrev PointedCone (𝕜 E) [OrderedSemiring 𝕜] [AddCommMonoid E] [Module 𝕜 E] :=
  Submodule {c : 𝕜 // 0 ≤ c} E


/-- Every pointed cone is a convex cone. -/
@[coe]
def toConvexCone (S : PointedCone 𝕜 E) : ConvexCone 𝕜 E where
  carrier := S
  smul_mem' c hc _ hx := S.smul_mem ⟨c, le_of_lt hc⟩ hx
  add_mem' _ hx _ hy := S.add_mem hx hy


instance : Coe (PointedCone 𝕜 E) (ConvexCone 𝕜 E) where
  coe := toConvexCone


theorem toConvexCone_injective : Injective ((↑) : PointedCone 𝕜 E → ConvexCone 𝕜 E) :=
                /-
                  𝕜 : Type u_1
                  E : Type u_2
                  inst✝² : OrderedSemiring 𝕜
                  inst✝¹ : AddCommMonoid E
                  inst✝ : Module 𝕜 E
                  x✝¹ x✝ : PointedCone 𝕜 E
                  ⊢ Eq ↑x✝¹ ↑x✝ → Eq x✝¹ x✝
                -/
  fun _ _ => by simp [toConvexCone]
                /-
                  🎉 no goals
                -/


@[simp]
theorem toConvexCone_pointed (S : PointedCone 𝕜 E) : (S : ConvexCone 𝕜 E).Pointed := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    inst✝² : OrderedSemiring 𝕜
    inst✝¹ : AddCommMonoid E
    inst✝ : Module 𝕜 E
    S : PointedCone 𝕜 E
    ⊢ (↑S).Pointed
  -/
  simp [toConvexCone, ConvexCone.Pointed]
  /-
    🎉 no goals
  -/


@[ext]
theorem ext {S T : PointedCone 𝕜 E} (h : ∀ x, x ∈ S ↔ x ∈ T) : S = T :=
  SetLike.ext h


instance instZero (S : PointedCone 𝕜 E) : Zero S :=
  ⟨0, S.zero_mem⟩


/-- The `PointedCone` constructed from a pointed `ConvexCone`. -/
def _root_.ConvexCone.toPointedCone {S : ConvexCone 𝕜 E} (hS : S.Pointed) : PointedCone 𝕜 E where
  carrier := S
  add_mem' hx hy := S.add_mem hx hy
  zero_mem' := hS
  smul_mem' := fun ⟨c, hc⟩ x hx => by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      S : ConvexCone 𝕜 E
      hS : S.Pointed
      x✝ : Subtype fun c => LE.le 0 c
      x : E
      hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
      c : 𝕜
      hc : LE.le 0 c
      ⊢ Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier (HS …
    -/
    simp_rw [SetLike.mem_coe]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝² : OrderedSemiring 𝕜
      inst✝¹ : AddCommMonoid E
      inst✝ : Module 𝕜 E
      S : ConvexCone 𝕜 E
      hS : S.Pointed
      x✝ : Subtype fun c => LE.le 0 c
      x : E
      hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
      c : 𝕜
      hc : LE.le 0 c
      ⊢ Membership.mem S (HSMul.hSMul ⟨c, hc⟩ x)
    -/
    cases' eq_or_lt_of_le hc with hzero hpos
      /-
        case inl
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        hS : S.Pointed
        x✝ : Subtype fun c => LE.le 0 c
        x : E
        hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
        c : 𝕜
        hc : LE.le 0 c
        hzero : Eq 0 c
        ⊢ Membership.mem S (HSMul.hSMul ⟨c, hc⟩ x)
      -/
    · unfold ConvexCone.Pointed at hS
      /-
        case inl
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        hS : Membership.mem S 0
        x✝ : Subtype fun c => LE.le 0 c
        x : E
        hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
        c : 𝕜
        hc : LE.le 0 c
        hzero : Eq 0 c
        ⊢ Membership.mem S (HSMul.hSMul ⟨c, hc⟩ x)
      -/
      convert hS
      /-
        case h.e'_5
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        hS : Membership.mem S 0
        x✝ : Subtype fun c => LE.le 0 c
        x : E
        hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
        c : 𝕜
        hc : LE.le 0 c
        hzero : Eq 0 c
        ⊢ Eq (HSMul.hSMul ⟨c, hc⟩ x) 0
      -/
      simp [← hzero]
      /-
        🎉 no goals
      -/
      /-
        case inr
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        G : Type u_4
        inst✝² : OrderedSemiring 𝕜
        inst✝¹ : AddCommMonoid E
        inst✝ : Module 𝕜 E
        S : ConvexCone 𝕜 E
        hS : S.Pointed
        x✝ : Subtype fun c => LE.le 0 c
        x : E
        hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
        c : 𝕜
        hc : LE.le 0 c
        hpos : LT.lt 0 c
        ⊢ Membership.mem S (HSMul.hSMul ⟨c, hc⟩ x)
      -/
    · apply ConvexCone.smul_mem
        /-
          case inr.hc
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          S : ConvexCone 𝕜 E
          hS : S.Pointed
          x✝ : Subtype fun c => LE.le 0 c
          x : E
          hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
          c : 𝕜
          hc : LE.le 0 c
          hpos : LT.lt 0 c
          ⊢ LT.lt 0 (↑Nonneg.coeRingHom.toMonoidWithZeroHom ⟨c, hc⟩)
        -/
      · convert hpos
        /-
          🎉 no goals
        -/
        /-
          case inr.hx
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          G : Type u_4
          inst✝² : OrderedSemiring 𝕜
          inst✝¹ : AddCommMonoid E
          inst✝ : Module 𝕜 E
          S : ConvexCone 𝕜 E
          hS : S.Pointed
          x✝ : Subtype fun c => LE.le 0 c
          x : E
          hx : Membership.mem { carrier := ↑S, add_mem' := ⋯, zero_mem' := hS }.carrier x
          c : 𝕜
          hc : LE.le 0 c
          hpos : LT.lt 0 c
          ⊢ Membership.mem S x
        -/
      · exact hx
        /-
          🎉 no goals
        -/


@[simp]
lemma _root_.ConvexCone.mem_toPointedCone {S : ConvexCone 𝕜 E} (hS : S.Pointed) (x : E) :
    x ∈ S.toPointedCone hS ↔ x ∈ S :=
  Iff.rfl


@[simp, norm_cast]
lemma _root_.ConvexCone.coe_toPointedCone {S : ConvexCone 𝕜 E} (hS : S.Pointed) :
    S.toPointedCone hS = S :=
  rfl


instance canLift : CanLift (ConvexCone 𝕜 E) (PointedCone 𝕜 E) (↑) ConvexCone.Pointed where
  prf S hS := ⟨S.toPointedCone hS, rfl⟩


/-- The image of a pointed cone under a `𝕜`-linear map is a pointed cone. -/
def map (f : E →ₗ[𝕜] F) (S : PointedCone 𝕜 E) : PointedCone 𝕜 F :=
  Submodule.map (f : E →ₗ[𝕜≥0] F) S


@[simp, norm_cast]
theorem toConvexCone_map (S : PointedCone 𝕜 E) (f : E →ₗ[𝕜] F) :
    (S.map f : ConvexCone 𝕜 F) = (S : ConvexCone 𝕜 E).map f :=
  rfl


@[simp, norm_cast]
theorem coe_map (S : PointedCone 𝕜 E) (f : E →ₗ[𝕜] F) : (S.map f : Set F) = f '' S :=
  rfl


@[simp]
theorem mem_map {f : E →ₗ[𝕜] F} {S : PointedCone 𝕜 E} {y : F} : y ∈ S.map f ↔ ∃ x ∈ S, f x = y :=
  Iff.rfl


theorem map_map (g : F →ₗ[𝕜] G) (f : E →ₗ[𝕜] F) (S : PointedCone 𝕜 E) :
    (S.map f).map g = S.map (g.comp f) :=
  SetLike.coe_injective <| Set.image_image g f S


@[simp]
theorem map_id (S : PointedCone 𝕜 E) : S.map LinearMap.id = S :=
  SetLike.coe_injective <| Set.image_id _


/-- The preimage of a convex cone under a `𝕜`-linear map is a convex cone. -/
def comap (f : E →ₗ[𝕜] F) (S : PointedCone 𝕜 F) : PointedCone 𝕜 E :=
  Submodule.comap (f : E →ₗ[𝕜≥0] F) S


@[simp, norm_cast]
theorem coe_comap (f : E →ₗ[𝕜] F) (S : PointedCone 𝕜 F) : (S.comap f : Set E) = f ⁻¹' S :=
  rfl


@[simp]
theorem comap_id (S : PointedCone 𝕜 E) : S.comap LinearMap.id = S :=
  rfl


theorem comap_comap (g : F →ₗ[𝕜] G) (f : E →ₗ[𝕜] F) (S : PointedCone 𝕜 G) :
    (S.comap g).comap f = S.comap (g.comp f) :=
  rfl


@[simp]
theorem mem_comap {f : E →ₗ[𝕜] F} {S : PointedCone 𝕜 F} {x : E} : x ∈ S.comap f ↔ f x ∈ S :=
  Iff.rfl


/-- The positive cone is the pointed cone formed by the set of nonnegative elements in an ordered
module. -/
def positive : PointedCone 𝕜 E :=
  (ConvexCone.positive 𝕜 E).toPointedCone <| ConvexCone.pointed_positive 𝕜 E


@[simp]
theorem mem_positive {x : E} : x ∈ positive 𝕜 E ↔ 0 ≤ x :=
  Iff.rfl


@[simp, norm_cast]
theorem toConvexCone_positive : ↑(positive 𝕜 E) = ConvexCone.positive 𝕜 E :=
  rfl


/-- The inner dual cone of a pointed cone is a pointed cone. -/
def dual (S : PointedCone ℝ E) : PointedCone ℝ E :=
  ((S : Set E).innerDualCone).toPointedCone <| pointed_innerDualCone (S : Set E)


@[simp, norm_cast]
theorem toConvexCone_dual (S : PointedCone ℝ E) : ↑(dual S) = (S : Set E).innerDualCone :=
  rfl


open scoped InnerProductSpace in
@[simp]
theorem mem_dual {S : PointedCone ℝ E} {y : E} : y ∈ dual S ↔ ∀ ⦃x⦄, x ∈ S → 0 ≤ ⟪x, y⟫_ℝ := by
  /-
    E : Type u_5
    inst✝¹ : NormedAddCommGroup E
    inst✝ : InnerProductSpace Real E
    S : PointedCone Real E
    y : E
    ⊢ Iff (Membership.mem S.dual y) (∀ ⦃x : E⦄, Membership.mem S x → LE.le 0 (Inne …
  -/
  rfl
  /-
    🎉 no goals
  -/


